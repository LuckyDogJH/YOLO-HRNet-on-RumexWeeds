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
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()  # Get absolute Path
ROOT = FILE.parents[0]  # File parent root path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path


import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
# from models.yolo import Model
from models.yolo_hrnet import yolo_hrnet
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)


## For distributed training, here -1 is default,because we use a single GPU. -1 disables DDP
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train(hyp, opt, device, callbacks):  # hyp : hyper-parameters path
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # save model directory
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'   #

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save settings in save directory
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance (by tensorBoard)

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Get Data INFO
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):   ## distributed training
        data_dict = check_dataset(data)  # check if data is None(./DataInfo.yaml)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'Weeds'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset


    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')

    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = yolo_hrnet(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # Initailize the model
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as Float32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = yolo_hrnet(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)

    # Freeze layer
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze, format: freeze = ['1', '8', '10' .....]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is grid_size_multiply

    # Estimate Batch_size if no Batch_size is givenn
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size if no batch_size is given
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    # optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    optimizer_yolov5 = smart_optimizer(model.model, opt.optimizer, hyp['lr0_yolov5'], hyp['momentum'], hyp['weight_decay'])
    optimizer_hrnet = smart_optimizer(model.hrnet_head, opt.optimizer, hyp['lr0_hrnet'], hyp['momentum'], hyp['weight_decay'])

    # Schedule learning rate
    if opt.cos_lr:
        lf_yolov5 = one_cycle(1, hyp['lrf_yolov5'], epochs)  # with epochs raising, result changes from: 1 -> hyp['lrf'], cosine curve
        lf_hrnet = one_cycle(1, hyp['lrf_hrnet'], epochs)
    else:
        lf_yolov5 = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf_yolov5']) + hyp['lrf_yolov5']  # linear line
        lf_hrnet = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf_hrnet']) + hyp['lrf_hrnet']  # linear line
    scheduler_yolov5 = lr_scheduler.LambdaLR(optimizer_yolov5, lr_lambda=lf_yolov5)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler_hrnet = lr_scheduler.LambdaLR(optimizer_hrnet, lr_lambda=lf_hrnet)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # # Resume
    best_fitness, start_epoch = 0.0, 0
    # if pretrained:
    #     if resume:
    #         best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
    #     del ckpt, csd     # delete for releasing memory

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,                # Train image folder or .txt
                                              imgsz,                     # image size
                                              batch_size // WORLD_SIZE,  # WORLD_SIZE = 1 in default
                                              gs,                        # stride
                                              single_cls,                # Default False(like what we do in Segmentation, we take multiple class as one class)
                                              hyp=hyp,                   # hyper-parameters
                                              augment=True,              # Data augmentation
                                              cache=None if opt.cache == 'val' else opt.cache,   # cache images in "ram" (default None) or "disk", Default None
                                              rect=opt.rect,             # rectangular training. Default False
                                              rank=LOCAL_RANK,           # for DDP model. default -1 -> nou usingDDP
                                              workers=workers,           # number of works in CPU
                                              image_weights=opt.image_weights,  # use weighted image selection, default False
                                              quad=opt.quad,             # quad dataloader, default False
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # check anchor size. if not sufficant, run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)


    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)

       # scale loss weights in case nl != 3
    hyp['box'] *= 3 / nl  # scale to box loss gain
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['kps'] *= 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers

    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup_iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1   # last optimization steps
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, valodation_loss(bbox, objectness_loss, classification_loss, kps_points_loss)
    scheduler_yolov5.last_epoch = start_epoch - 1
    scheduler_hrnet.last_epoch = start_epoch - 1

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    ## Early Stopping
    stopper, stop = EarlyStopping(patience=opt.patience), False

    compute_loss = ComputeLoss(model, roi_output_size=opt.roi_output_size)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    ### ----------------------------------------- Begin training ------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights according to mAP per class (optional, single-GPU only), default False
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # sample image in dataset according to image weights

        mloss = torch.zeros(4, device=device)  # mean losses

        if RANK != -1:   # for DDP
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 8) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'kps_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer_yolov5.zero_grad()
        optimizer_hrnet.zero_grad()

        ## ---------------------------------------- Do Batches in Epoch -------------------------------------------------
        for i, (imgs, targets, HeatMaps, paths, _) in pbar:   # (images, label, self.im_files, image_shape)
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number of batches since training start
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 从 1 升到 [64/batch_size]
                for j, x in enumerate(optimizer_yolov5.param_groups):
                    # optimizer_groups: 1.bias in BN  2. weights in nn.Conv2d   3.weights in BN
                    # bias lr falls from 0.1 to lr0, all the other lr rises from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf_yolov5(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                for j, x in enumerate(optimizer_hrnet.param_groups):
                    # optimizer_groups: 1.bias in BN  2. weights in nn.Conv2d   3.weights in BN
                    # bias lr falls from 0.1 to lr0, all the other lr rises from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf_hrnet(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale(default False)
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):  ## amp training
                pred, loss, loss_items = model(imgs, targets.to(device), {key: HeatMaps[key].to(device) for key in HeatMaps.keys()}, compute_loss)  # forward
                # loss(loss for a whole batch): (lbox + lobj + lcls + lkps)
                # loss_items: [lbox, lobj, lcls, lkps]
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimizer update - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:

                ## can also do not use this below two lines
                scaler.unscale_(optimizer_yolov5)  # unscale gradients
                scaler.unscale_(optimizer_hrnet)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients

                scaler.step(optimizer_yolov5)  # optimizer.step()
                scaler.step(optimizer_hrnet)  # optimizer.step()
                scaler.update()
                optimizer_yolov5.zero_grad()
                optimizer_hrnet.zero_grad()

                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr_yolov5 = [x['lr'] for x in optimizer_yolov5.param_groups]  # for loggers
        scheduler_yolov5.step()

        lr_hrnet = [x['lr'] for x in optimizer_hrnet.param_groups]  # for loggers
        scheduler_hrnet.step()
        print(f'yolov5 learning rate{lr_yolov5[0]} \nhrnet learning rate{lr_hrnet[0]} ')

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss,
                                                roi_output_size=opt.roi_output_size,
                                                root_conf_thresh=opt.root_conf_thresh)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95, OKS]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            # callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer_yolov5': optimizer_yolov5.state_dict(),
                    'optimizer_hrnet': optimizer_hrnet.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).float(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        roi_output_size=opt.roi_output_size,
                        root_conf_thresh=opt.root_conf_thresh)  # val best model with plots

    torch.cuda.empty_cache()
    return results



def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()   ## Check if code is updated
        check_requirements() ## Check if python dependency is installed

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():      # if opt_yaml exists
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks data, model_configuration, hyper-parameters, saved_model_weights, prediction_file_save_path
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # if opt.evolve:   # re-write save_path, this is for auto-tuning hyperparameters  training
        #     if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
        #         opt.project = str(ROOT / 'runs/evolve')
        #     opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        # if opt.name == 'cfg':
        #     opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    ## --------------------------------------------- Train ------------------------------------------------------------
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)



def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


def parse_opt(known=False):
    '''
    num_class: number of class
    weigihts: pre-trained model
    cfg: model architecture if we want train from beginning without using pre-trained model, you can set to './models/yolov5l.yaml'
    data: Dataset Info file
    hyp: hyper-parameters file
    epochs: number of epochs
    batch_size
    imgsz: image size
    rect: if rectanglar training, action='store true' means when using rect, we only need to do: ！python train.py --rect, 不用加=
    resume: train from recent training
    nosave: only save final checkpoint
    noval: only validate final epoch
    noautoanchor: autoanchor: if your anchor is not reasonable, it will calculate the anchor based on you GT. noautoanchor: disable the function
    evolve: if we donot know how to adjust the hyper-parameters, we can usu evolve to automatically try with different hyp
    bucket
    cache: 在加载数据时，在ram中保存第一个epoch的数据的中间计算，这样在后来的epoch可以减小计算量
    image_weights: 应该是只在classify中用
    multi-scale
    single-cls: merge many class to one class
    optimizer:
    sync-bn: 在DDP中是否同步多卡的batch-normalization
    workers
    project： save path for model and results during process
    name: save to ./project/name
    exist_ok: if exist ./project/name, overlap it
    quad
    cos_lr: 余弦退火的lr
    label_smoothing
    patience: early stop when no improvement happens after patience epochs
    freeze: freeze layers
    save_period: save checkpoints every x epochs
    loacl_rank: DDP parameter
    ./yolov5m.pt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--weights', type=str, default='./yolov5m.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./DataInfo.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='./hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', default=True, help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='./runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--roi_output_size', type=int, default=16)
    parser.add_argument('--root_conf_thresh', type=float, default=0.0)

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

