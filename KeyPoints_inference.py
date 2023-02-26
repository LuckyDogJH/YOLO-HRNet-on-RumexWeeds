# Author: Jiahao Li
# CreatTime: 2022/12/5
# FileName: 
# Description: None

import torch
import torch.nn.functional as F
import torchvision


def clip_boundary(info, refer_img):
    info[:, [0, 2]] = info[:, [0, 2]].clamp_(0, refer_img.size(2))
    info[:, [1, 3]] = info[:, [1, 3]].clamp_(0, refer_img.size(1))
    return info

def map_to_bbox(bbox_info, current_stride, HeatMap):
    bbox_info *= current_stride
    w, h = max(int(bbox_info[2]-bbox_info[0]), 2), max(int(bbox_info[3] - bbox_info[1]), 2) # w, h
    HeatMap = F.interpolate(HeatMap[None, None, ...], (h, w), mode='bilinear').squeeze() # Input must be [B, C ,H, W]
    y, x = torch.where(HeatMap == torch.max(HeatMap))
    x, y = bbox_info[0] + x, bbox_info[1] + y
    return x[0], y[0]


def KeyPoints_inference(image_ind, pred, feature_maps, hrnet, stride, roi_out_size=8, maxValue_thresh=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_pred = torch.zeros((pred.size(0), pred.size(1)+2), dtype=torch.float32, device=device)
    for i in range(0, 3):
        begin_ind = 0
        index = pred[:, -1] == float(i)
        outputs = torch.zeros((pred[index].size(0), 2), dtype=torch.float32, device=device)
        input_info = pred[index].clone().to(device)
        if input_info.size(0) == 0:
            continue
        current_stride = stride[i]
        current_feature_map = feature_maps[i][image_ind]
        input_info = input_info[:, :4]/current_stride
        # expand bbox
        # input_info[:, [0, 1]] -= input_info[:, [0, 1]] * 0.1
        # input_info[:, [2, 3]] += input_info[:, [2, 3]] * 0.1
        input_info = clip_boundary(input_info, current_feature_map)


        # ROI Align
        current_feature_map = current_feature_map[None, ...]
        roi_feature_map = torchvision.ops.roi_align(current_feature_map.to(torch.float).to(device), [input_info], output_size=roi_out_size, sampling_ratio=2)
        hrnet_pred = hrnet._hrnet_forward_once(roi_feature_map, i)
        HeatMap = torch.sigmoid(hrnet_pred.squeeze(1))
        assert len(HeatMap.size()) == 3, f'HeatMap size is {HeatMap.size()}, should be BatchSize,{roi_out_size, roi_out_size}'


        # Map Keypoints to bbox
        ## Filter out low scores
        HeatMap_max = torch.max(HeatMap.view(HeatMap.size(0), -1), 1)[0]
        Has_root_indices = torch.where(HeatMap_max > maxValue_thresh)[0]
        for j in range(Has_root_indices.size(0)):
            current_ind = begin_ind + Has_root_indices[j]
            cx, cy = map_to_bbox(input_info[Has_root_indices[j]], current_stride, HeatMap[Has_root_indices[j]])
            outputs[current_ind, 0], outputs[current_ind, 1] = cx, cy
        final_pred[index] = torch.cat((pred[index], outputs), 1)
    return final_pred