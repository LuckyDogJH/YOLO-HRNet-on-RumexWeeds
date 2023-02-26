# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s.xml                # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path
from KeyPoints_inference import KeyPoints_inference

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_boxes_for_pred, scale_boxes_for_target,xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, compute_OKS
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls, cxcy in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf, *cxcy) if save_conf else (cls, *xywh, *cxcy)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # convert xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'root_xy': [round(x, 3) for x in p[7:]],
            'confidence': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 9]), num_pred_bbox, [x_min, y_min, x_max, y_max, conf, class, feature_map_ind, root_x, root_y]
        labels (array[M, 8]), num_GT_bbox, [class, x_min, y_min, x_max, y_max, root_x, root_y, root_class]
        iouv (array [1, 10]), torch.linspace(0.5, 0.95, 10)
    Returns:
        correct (array[N, 10]), for 10 IoU_thresh levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:5], detections[:, :4])  # calculate the iou of each pred bbox with each GT bbox
    correct_class = labels[:, 0:1] == detections[:, 5] # check the pred class alignment with GT class
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            match_info = torch.from_numpy(matches[:, :2]) if x[0].shape[0] else torch.zeros((0, 2), dtype=torch.float32)
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), match_info.to(torch.int)


@smart_inference_mode()
def run(
        data,       # data path and train.txt, val.txt path
        weights=None,  # model.pt path
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half = False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        root_conf_thresh=0.1,
        roi_output_size=10
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()   # number of iou : 10

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95', 'OKS')
    tp, fp, p, r, f1, mp, mr, mAP50, ap50, mAP, OKS = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile(), Profile()
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class, matched_keyPoints = [], [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, HeatMaps, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out, feature_maps = model(im, targets, HeatMaps, compute_loss) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            mloss, loss_item = compute_loss(model.hrnet_head.float(), train_out, targets, HeatMaps, feature_maps)
            loss += loss_item  # box, obj, cls, oks

        # NMS
        targets[:, [2, 3, 4, 5, 7, 8, 9, 10]] *= torch.tensor((width, height, width, height, width, height, width, height), device=device)  # 去归一化
        #target.size() = number_of_total_objects_whole_batch * 11(img_index_in_batch, category, bx, by, bw, bh, root_class, cx, cy, cw, ch)

        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=False,
                                        agnostic=single_cls,
                                        max_det=max_det,
                                        nm=1)
            # preds.size() = batch_size, num_pred_bbox_after_nms, 7(x_min, y_min. x_max, y_max, final_score, category_label, feature_map_index)
            # target.size() ==  number_of_total_objects_whole_batch * 11(img_index_in_batch, category, bx, by, bw, bh, root_class, cx, cy, cw, ch)


        # mAP Metrics and KeyPoints Inference
        preds_with_root = []
        for si, pred in enumerate(preds):

            ############################################################################################################
            #########################################Inference KeyPoints################################################
            ############################################################################################################

            pred = KeyPoints_inference(si, pred, feature_maps, model.hrnet_head, model.stride, roi_out_size=roi_output_size, maxValue_thresh=root_conf_thresh)
            ## Now the pred becomes
            ## pred.size() = num_bbox_after_nms , 9(xmin, ymin, xmax, ymax, conf, category, featuremap_index, cx, cy)


            pred_with_root = pred.clone()
            # pred_with_root[:, [0, 2]], pred_with_root[:, [1, 3]] = pred_with_root[:, [0, 2]].clamp_(0, shapes[si][0][1]), pred_with_root[:, [1, 3]].clamp_(0, shapes[si][0][0])
            pred_with_root[:, [0, 2]], pred_with_root[:, [1, 3]] = pred_with_root[:, [0, 2]].clamp_(0, im.size(-1)), pred_with_root[:, [1, 3]].clamp_(0, im.size(-2))
            preds_with_root.append(pred_with_root)
            ############################################################################################################
            ############################################################################################################
            ############################################################################################################

            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]  # path: image_path, shape: orig_image_shape
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init, npr: num_pred_bbox, niou: 10(mAP from 0.5 to 0.95)
            seen += 1   # for calculate time consuming

            if npr == 0:
                if nl:
                    stats.append((correct.cpu(), *torch.zeros((2, 0), device=device).cpu(), labels[:, 0].cpu()))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue
            
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn = scale_boxes_for_pred(im[si].shape[1:], predn, shape, shapes[si][1])
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, [1, 2, 3, 4, 6, 7, 5]])  # xywh -> x_min, y_min, x_max, y_max
                # tbox = x_min, y_min, x_max, y_max, root_x, root_y, root_class

                tbox = scale_boxes_for_target(im[si].shape[1:], tbox, shape, shapes[si][1])  # Rescale boxes (xyxy) and (root_x, root_y) to original_img_shape
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct, match_info = process_batch(predn.to(device), labelsn.to(device), iouv)
                if plots:
                    confusion_matrix.process_batch(predn[:, :6].cpu(), labelsn[:, :5].cpu())
                if match_info.size(0):
                    GT_w = (labelsn[match_info[:, 0][0], [3]] - labelsn[match_info[:, 0][0], [1]]).view(-1, 1).cpu()
                    GT_h = (labelsn[match_info[:, 0][0], [4]] - labelsn[match_info[:, 0][0], [2]]).view(-1, 1).cpu()
                    GT_area = (GT_w * GT_h * 0.5).cpu()
                    GT_root_class = labelsn[match_info[:, 0][0], [-1]].view(-1, 1).cpu()
                    GT_root_position = labelsn[match_info[:, 0][0], [5, 6]].view(-1, 2).cpu()
                    pred_root_position = predn[match_info[:, 1][0], [-2, -1]].view(-1, 2).cpu()
                    matched_keyPoints.append(torch.cat((GT_root_class, GT_area, GT_root_position, pred_root_position), 1))
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))  # (correct, conf, pred_class, target_class)
            # stats.append((correct.cpu(), aa[:, 4].cpu(), aa[:, 5].cpu(), labels[:, 0].cpu()))  # (correct, conf, pred_class, target_class)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds_with_root), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred
        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute mAP metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, mAP50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Compute OKS
    # matched_KeyPoints.size() = num_matched_GT, 6(GT_root_class, GT_area, GT_root_x, GT_root_y, pred_root_x, pred_root_y)
    if len(matched_keyPoints):
        matched_keyPoints = torch.cat(matched_keyPoints, 0)
        OKS = compute_OKS(matched_keyPoints, sigma=0.2)
    else:
        OKS = 1000.0

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, mAP50, mAP, OKS))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')
    if OKS == 1000.0:
        LOGGER.warning(f'WARNING ⚠️ no matched Bbox found in {task} set, can not compute OKS without matching')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], OKS))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            mAP, mAP50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    mAPs = np.zeros(nc) + mAP
    for i, c in enumerate(ap_class):
        mAPs[c] = ap[i]
    return (mp, mr, mAP50, mAP, OKS, *(loss.cpu() / len(dataloader)).tolist()), mAPs, t





