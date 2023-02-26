# Author: Jiahao Li
# CreatTime: 2022/12/15
# FileName:
# Description: None


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch(beginning)

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import sys
from pathlib import Path
import cv2

import numpy as np
import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()  # Get absolute Path
ROOT = FILE.parents[0]  # File parent root path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path


from KeyPoints_inference import KeyPoints_inference
from val import non_max_suppression, process_batch # for end-of-epoch mAP
from models.yolo_hrnet import yolo_hrnet
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, colorstr, scale_boxes_for_pred,
                           scale_boxes_for_target, xywh2xyxy, xyxy2xywh_1)
from utils.metrics import ap_per_class, compute_OKS
from utils.plots import Colors, Annotator


def output_to_target(output, max_det=300):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf, root_x, root_y] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, _, roots = o[:max_det].cpu().split((4, 1, 1, 1, 2), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh_1(box), conf, roots), 1))
    return torch.cat(targets, 0).numpy()



def plot_images(images, targets, paths=None, fname='images.jpg', names=None):
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    colors = Colors()
    max_size = 1920  # max image size
    max_subplots = 1  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width(size of training image)
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders for each image
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # put image_name on the top of each image
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            roots = ti[:, 7:]
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 11  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                    roots[:, 0] *= w
                    roots[:, 1] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
                    roots *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            roots[:, 0] += x
            roots[:, 1] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = (255, 215, 0)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
                    if roots[j].any() > 0.0:
                        annotator.circle(roots[j], boundary=[x, y, x + w, y + h], r=4, fill=(0, 204, 255))
    annotator.im.save(fname)  # save


def inference(
        model,
        data,
        dataloader,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        roi_output_size=10,
        root_conf_thresh=0.1,
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=True,  # treat as single-class dataset
        augment=False,  # augmented inference
        compute_loss=None,
        names = None
    ):
    model.float()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()   # number of iou : 10

    seen = 0
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95', 'OKS')
    tp, fp, p, r, f1, mp, mr, mAP50, ap50, mAP, OKS = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile(), Profile()
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class, matched_keyPoints = [], [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    with torch.no_grad():
        for batch_i, (im, targets, HeatMaps, paths, shapes) in enumerate(pbar):
            with dt[0]:
                if cuda:
                    im = im.to(device, non_blocking=True)
                    targets = targets.to(device)
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                preds, train_out, feature_maps = model(im, targets, HeatMaps, compute_loss)



            # NMS
            targets[:, [2, 3, 4, 5, 7, 8, 9, 10]] *= torch.tensor((width, height, width, height, width, height, width, height), device=device)  # 去归一化
            #target.size() = number_of_total_objects_whole_batch * 11(img_index_in_batch, category, bx, by, bw, bh, root_class, cx, cy, cw, ch)
            with dt[2]:
                preds = non_max_suppression(preds,
                                            conf_thres,
                                            iou_thres,
                                            labels=[],
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
                with dt[3]:
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
                    # labelsn = category, x_min, y_min, x_max, y_max, root_x, root_y, root_class

                    correct, match_info = process_batch(predn.to(device), labelsn.to(device), iouv)
                    if match_info.size(0):
                        GT_w = (labelsn[match_info[:, 0][0], [3]] - labelsn[match_info[:, 0][0], [1]]).view(-1, 1).cpu()
                        GT_h = (labelsn[match_info[:, 0][0], [4]] - labelsn[match_info[:, 0][0], [2]]).view(-1, 1).cpu()
                        GT_area = (GT_w * GT_h * 0.5).cpu()
                        GT_root_class = labelsn[match_info[:, 0][0], [-1]].view(-1, 1).cpu()
                        GT_root_position = labelsn[match_info[:, 0][0], [5, 6]].view(-1, 2).cpu()
                        pred_root_position = predn[match_info[:, 1][0], [-2, -1]].view(-1, 2).cpu()
                        matched_keyPoints.append(torch.cat((GT_root_class, GT_area, GT_root_position, pred_root_position), 1))
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))  # (correct, conf, pred_class, target_class)

            # Plot images
            image_name = '.'.join([paths[0].split('.')[0].split('/')[-1], 'png'])
            plot_images(im, targets, None, os.path.join('./results/GroundTruth', image_name), names)  # labels
            plot_images(im, output_to_target(preds_with_root), None, os.path.join('./results/Prediction', image_name), names)

    # Compute mAP metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names=names)
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
        LOGGER.warning(f'WARNING ⚠️ no labels found in testset, can not compute metrics without labels')
    if OKS == 1000.0:
        LOGGER.warning(f'WARNING ⚠️ no matched Bbox found in testset, can not compute OKS without matching')

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], OKS))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms key-point inference time per image at shape {shape}' % t)

    # Plots
    mAPs = np.zeros(nc) + mAP
    for i, c in enumerate(ap_class):
        mAPs[c] = ap[i]

def main(opt):  # hyp : hyper-parameters path

    if not os.path.exists(opt.results):
        os.mkdir(opt.results)
        os.mkdir(os.path.join(opt.results, 'GroundTruth'))
        os.mkdir(os.path.join(opt.results, 'Prediction'))

    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints


    data_dict = check_dataset(opt.data)  # check if data is None(./DataInfo.yaml)
    test_path = data_dict['test']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'Weeds'} if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(opt.model, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = yolo_hrnet(ckpt['model'].yaml, ch=3, nc=nc, anchors=None)  # Initailize the model
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as Float32
    model.load_state_dict(csd)
    model.hyp = hyp
    model.to(device)


    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is grid_size_multiply
    test_loader = create_dataloader(test_path,
                                   imgsz,
                                   opt.batch_size,
                                   gs,
                                   opt.single_cls,
                                   cache=None,
                                   rect=True,
                                   rank=-1,
                                   pad=0.0,
                                   prefix=colorstr('test: '))[0]

    model.eval()
    inference(
        model,
        data_dict,
        test_loader,
        batch_size=opt.batch_size,
        imgsz=imgsz,
        conf_thres=opt.conf_thresh,
        iou_thres=0.60,
        max_det=300,
        roi_output_size=opt.roi_output_size,
        root_conf_thresh=opt.root_conf_thresh,
        single_cls=opt.single_cls,
        compute_loss=None,  # Only for start up the model, but will not be used here
        names=names
        )

    torch.cuda.empty_cache()





def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--model', type=str, default='./saved_model/best.pt')
    parser.add_argument('--data', type=str, default='./DataInfo.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='./hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', default=True, help='train multi-class data as single-class')
    parser.add_argument('--results', type=str, default='./results')
    parser.add_argument('--roi_output_size', type=int, default=16)
    parser.add_argument('--root_conf_thresh', type=float, default=0.0)
    parser.add_argument('--conf_thresh', type=float, default=0.001)
    parser.add_argument('--iou_thresh', type=float, default=0.6)

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

