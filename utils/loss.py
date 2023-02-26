# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Initialize Focal Loss:  criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma        # for easy or hard sample
        self.alpha = alpha        # for positive or negative sample
        self.reduction = loss_fcn.reduction  # focal_loss 使用BCE中的reduction方法：'mean'
        self.loss_fcn.reduction = 'none'     # 取均值只在focal_loss中取一次就够了，BCE不要提前算mean

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # get probility by sigmod，（0， 1）

        # 预测正确的概率 p_t
        # label=1, 预测正确的概率就是pred_prob; label=0时，因为正确的label是0，预测正确的概率是1-pred_prob
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)

        # Balance positive and negative samples
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)

        # Distinguish easy or hard samples
        # 不管label是1还是0， 难易程度的区分都是根据当前label的对立面计算的，即1-p_t
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, roi_output_size=10):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        MSEkps = nn.MSELoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # get the Detect() module
        # Set weights for obj loss in three detect layers. weights of 80 X 80 is the biggest because small object detection is applied on this layer, which is a difficult task
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance_kps = {3: [4.0, 2.0, 1.0]}.get(m.nl)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.MSEkps, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, MSEkps, 1.0, h, autobalance  # self.gr: confidence ratio, utilized below
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.roi_output_size = roi_output_size


    def xywh2xyxy(self, bbox, img_shape):
        (x1, y1, w1, h1) = bbox.chunk(4, 1)
        w1_, h1_ = w1 / 2, h1 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        return torch.cat((b1_x1.clamp_(0, img_shape[1]-1), b1_y1.clamp_(0, img_shape[0]-1), b1_x2.clamp_(0, img_shape[1]-1), b1_y2.clamp_(0, img_shape[0]-1)), 1)

    def __call__(self, hrnet_head, p, targets, HeatMaps, feature_maps):  # predictions, targets
        # p:list, p.size() = num_layers(3)*[ batch_size,num_anchor_per_featuremap , h, w, (num_class + 5)]  e.g. 3 * 16 * 3 * 80 * 80 * 7
        # targets.size() = num_object_in_all_images, 6(img_index_in_batch, category, x, y, w, h)
        lcls = torch.zeros(1, device=self.device)  # class loss initialize
        lbox = torch.zeros(1, device=self.device)  # box loss initialize
        lobj = torch.zeros(1, device=self.device)  # object loss initialize
        lkps = torch.zeros(1, device=self.device)  # KeyPoints loss initialize

        # Assign the Positive Samples of target bboxes
        tcls, tbox, indices, anchors, grid_coord = self.build_targets(p, targets[:, :6]) # tcls: class index
                                                                      # tbox: box_info [x_distance_to_top-left_cell-corner, y_distance_to_top-left_cell-corner, box_w, box_h]
                                                                      # indices: image_index, anchor_index, grid_cell_top-left_y, grid_cell_top-left_x
                                                                      # anchors: width and height of anchors
                                                                      # grid_coord: Grid top-left corner for GT bbox

        # Losses
        for i, pi in enumerate(p):  # layer index, predictions per detection layer
            b, a, gj, gi = indices[i]  # image_index, anchor_index, grid_cell_top-left_y, grid_cell_top-left_x

            # initialze objectness label for both positive and negative samples
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # batch_size, num_anchors, h, w,

            n = b.shape[0]  # number of positive samples
            if n:
                # 取出正样本区域对应的预测结果(n * (5 + num_class))后再切分
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # tensor_split(list, dim)): 沿着dim方向切分成[:2], [2:4], [4:5], [5:]四块
                                                                                     # [x y], [w h], confidence, [each class score]

                pxy = pxy.sigmoid() * 2 - 0.5    # yolov5 与 yolov4 对 cx， cy加上scale的策略来使bbox中心点距离grid cell 左上角的偏移量在[-0.5, 1.5]
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]   # yolov5 对于bw， bh的计算方法
                pbox = torch.cat((pxy, pwh), 1)  # predicted box_info

        ##########################################################################################################################################
        ##########################################################################################################################################
        ###################################                                                                    ###################################
        ###################################           HRnet-KeyPoint Detection Part Start                      ###################################
        ###################################                                                                    ###################################
        ##########################################################################################################################################
        ##########################################################################################################################################

                grid_corner = grid_coord[i]   # 每个正样本对应的grid cell的coordinates
                feature_map = feature_maps[i] # get current feature map
                HeatMap = HeatMaps[str(i)].unsqueeze(1)   # get current heat map, add a dimension cuz roi_align requires [B, C, H, W]
                # 从feature map上裁剪pred_bbox部分
                ## 得到对应于特征图中的x_min, y_min, x_max, y_max(w, h会扩大一些)
                pbox_for_cut = pbox.clone().detach()
                pbox_for_cut[:, :2] = pbox_for_cut[:, :2] + grid_corner
                pbox_for_cut[:, 2:] = pbox_for_cut[:, 2:] * 1.2   # 略微扩大pred bbox的w, h
                pbox_for_cut = self.xywh2xyxy(pbox_for_cut, tobj.size()[2:])

                ## 把每个bbox的image index加上
                pbox_for_cut_with_imgInd = torch.cat((b[..., None], pbox_for_cut), 1)

                ## ROI Align
                roi_feature_map = torchvision.ops.roi_align(feature_map.to(torch.float32), pbox_for_cut_with_imgInd.to(torch.float32), output_size=self.roi_output_size, sampling_ratio=2)
                roi_HeatMap = torchvision.ops.roi_align(HeatMap.to(self.device), pbox_for_cut_with_imgInd, output_size=self.roi_output_size, sampling_ratio=2)

                ## HRnet Predict KeyPoints
                if roi_feature_map.size(0) > 15:
                    hrnet_res = []
                    begin_ind = 0
                    while begin_ind+15 < roi_feature_map.size(0):
                        hrnet_pred = hrnet_head._hrnet_forward_once(roi_feature_map[begin_ind:begin_ind+15].to(self.device), i)
                        hrnet_res.append(hrnet_pred)
                        begin_ind += 15
                    hrnet_pred = hrnet_head._hrnet_forward_once(roi_feature_map[begin_ind:].to(self.device), i)
                    hrnet_res.append(hrnet_pred)
                    hrnet_pred = torch.cat(hrnet_res, 0).to(self.device)
                else:
                    hrnet_pred = hrnet_head._hrnet_forward_once(roi_feature_map.to(self.device), i)

                ## KPS Loss(MSE)
                lkps += self.MSEkps(torch.sigmoid(hrnet_pred), roi_HeatMap.to(self.device)) * self.balance_kps[i]
        ##########################################################################################################################################
        ##########################################################################################################################################
        ###################################                                                                    ###################################
        ###################################           HRnet-KeyPoint Detection Part  End                       ###################################
        ###################################                                                                    ###################################
        ##########################################################################################################################################
        ##########################################################################################################################################

                # Regression(iou loss for box loss)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # 启用DIoU，GIoU 或者 CIoU的话，iou<=0 代表无重合的地方，iou.max=1, iou.min=-1
                lbox += (1.0 - iou).mean()  # ciou loss

                # Objectness(BCE loss or could be focal loss)
                iou = iou.detach().clamp(0).type(tobj.dtype)  # 去掉iou<=0的部分，该部分是无重合的情况

                # ## 如果检测目标较为聚集，开启self.sort_obj_iou比较好
                # if self.sort_obj_iou:
                #     j = iou.argsort()
                #     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]


                # 因为target的bbox没有置信度，因此需要人为给定一个confidence的label来计算objectness loss -> self.gr: confidence ratio, [0, 1]
                # self.gr 越接近0， confidence的label越接近1，适合训练更难区分的样本. 默认self.gr = 1
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # set the objectness label of positive samples to iou(0~1)

                # Classification (BCEloss -> or could be focal loss)
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # intialize one-hot category label
                    t[range(n), tcls[i]] = self.cp       # put category index in one-hot label
                    lcls += self.BCEcls(pcls, t)     # BCE loss or focal loss

            ## 计算所有正样本与负样本的 objectness loss, tobj中，负样本label为0
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # weighted objectness loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkps *= self.hyp['kps']
        bs = tobj.shape[0]  # batch size
        return (lbox + lobj + lcls + lkps) * bs, torch.cat((lbox, lobj, lcls, lkps)).detach()

    def build_targets(self, p, targets):
        # p:list, p.size() =  num_layers(i.e. 3), batch_size, num_anchor_per_featuremap , h, w, (num_class + 5)  e.g. 3 * 16 * 3 * 80 * 80 * (1 + 5)
        # targets.size() = num_object_in_all_images, 6(img_index_in_batch, category, x, y, w, h) <- [normalized]

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, num_targets(number of all the objects in one batch)
        tcls, tbox, indices, anch, grid_coord = [], [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # ai.size() = num_anchors, num_objects_in_whole_batch
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # a.repeat(n, 1, 1): 表示在相应维度上重复n倍， 1则表示不变
                                                                           # a[..., None]: 在最后加一个维度, ...表示所有维度， ：只代表一个维度，
                                                                           # a[..., None] == a[:,:,None] (a is 2-d )
                                                                           # 对每一个targets， 都分配给num_anchor个anchor
        # Now, targets.size() = num_anchors(3), num_objects, 7, 7表示：[img_index, category_index, cx, cy, w, h, anchor_index] <- [normalized]


        for i in range(self.nl):  # num_layers: 在 each feature map上
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # tensor[[3,2,3,2]] 得到tensor中 3th, 2th, 3th, 2th 并组成新的tensor
                                                           # [feature_map_w, map_h, map_w, map_h]
            # gain = [1, category_index, feature_map_w, ~map_h, ~map_w, ~map_h, 1]

            # Positive and Negative sample assignment
            t = targets * gain  # shape(3,n,7), cx, cy, w, h become [un-normalized]
            if nt:
                # Positive and Negative sample
                ## 计算target和每一个anchor(一共3个)的宽高比
                r = t[..., 4:6] / anchors[:, None]  # ratio of width and height
                ## 如果宽高比小于self.hyp['anchor_t']这个阈值， 则划分为正样本
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare ratio with threshold, max_thresh = 4
                                                                          # 最终把预测的bw， bh转到实际图像是 ： w = Pw * ((2*sigmoid(bw))^2), 最大就是anchor size的4倍
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # Positive sample, size = [num_positive_samples, 7]

                # Offsets
                # YOLOv5样本划分策略，若一个网格中的center距离网格边缘小于g(0.5), 那么临近的网格(仅限正上、下、左、右)也可以被划分为正样本
                g = 0.5  # bias
                off = torch.tensor(
                    [
                        [0, 0],   # current grid cell
                        [1, 0],   # left_near cell
                        [0, 1],   # top_near cell
                        [-1, 0],  # right_near cell
                        [0, -1],  # bottom_near cell
                    ], device=self.device).float() * g

                gxy = t[:, 2:4]   # center_x, center_y, distance to left
                gxi = gain[[2, 3]] - gxy  # inverse, distance to right
                # j, k, l, m: left, top, right, bottom
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # stack 与 cancatanate: cancatanate沿着dim融合，不产生新维度： e.g: (2,3) + (2.3) -> (4,3)
                                                                  #                       stack沿着dim拼接，仍保留维度    e.g. (2.3) + (2.3) -> (2, 2, 3)


                ## 把正样本再分成相同的5份，分别代表 current_cell, left_cell, top_cell, right_cell, bottom_cell, 根据jklm的情况选取正样本
                t = t.repeat((5, 1, 1))[j]   # size = [num_all_positive_samples, 7]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # assign the top-left corrdinates of cell for each positive sample according to offset
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image_index, class_index), (center_x, center_y), (center_w, center_h), anchor_index
                                             # tensor.chunk(m, n) : 沿着dim=n把tensor分成m块
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()   # get the top-left coordinates of cell as (center_x, center_y)
            gi, gj = gij.T

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image_index, anchor_index, grid_cell_top-left_y, grid_cell_top-left_x
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box_info: x_distance_to_top-left_cell-corner, y_distance_to_top-left_cell-corner, w, h
            anch.append(anchors[a])  # width and height of anchors
            tcls.append(c)  # class_index
            grid_coord.append(gij)

        return tcls, tbox, indices, anch, grid_coord
