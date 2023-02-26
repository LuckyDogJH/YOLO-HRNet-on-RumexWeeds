# Author: Jiahao Li
# CreatTime: 2022/12/1
# FileName: 
# Description: None


import numpy as np
import torch
import cv2


class KeypointToHeatMap(object):
    def __init__(self, gaussian_sigma_thresh: int = 2):
        self.sigma = gaussian_sigma_thresh


    def __call__(self, labels, orig_img_size,stride=[8, 16, 32]):
        heatmap_hw = np.array([orig_img_size / i for i in stride], dtype=np.uint8)
        if len(labels):
            Heatmaps = []
            kps_info = labels[:, 5:]
            num_kps = kps_info.shape[0]
            for i in range(len(heatmap_hw)):
                heatmap_size = heatmap_hw[i]
                heatmap = np.zeros((heatmap_size[0], heatmap_size[1]), dtype=np.float32)
                for kps_id in range(num_kps):
                    if kps_info[kps_id][0] == 0:
                        continue
        
                    # Use np.ceil in case zero occurs
                    cx, cy = int(np.ceil(kps_info[kps_id][1] * heatmap_size[1])), int(np.ceil(kps_info[kps_id][2] * heatmap_size[0]))
                    rx, ry = int(np.ceil(kps_info[kps_id][3] * heatmap_size[1])), int(np.ceil(kps_info[kps_id][4] * heatmap_size[0]))
                    radius_x, radius_y = min(rx, self.sigma * 3), min(ry, self.sigma * 3)
                    up_left, bottom_right = [cx - radius_x, cy - radius_y], [cx + radius_x, cy + radius_y]
                    radius = [radius_x, radius_y]
        
                    # if edge points beyond boundary, ignore these points
                    if (up_left[0] > heatmap_size[1] - 1) or (up_left[1] > heatmap_size[0] - 1) or (bottom_right[0] < 0) or (bottom_right[1] < 0):
                        continue
        
                    # generate gaussian kernel
                    kernel = self.gaussian_kernel(radius, sigma=self.sigma)
        
                    # Find the range in kernel
                    g_x = (max(0, -up_left[0]), min(bottom_right[0], heatmap_size[1] - 1) - up_left[0])
                    g_y = (max(0, -up_left[1]), min(bottom_right[1], heatmap_size[0] - 1) - up_left[1])
        
                    # Find the range in Heatmap
                    img_x = (max(0, up_left[0]), min(bottom_right[0], heatmap_size[1] - 1))
                    img_y = (max(0, up_left[1]), min(bottom_right[1], heatmap_size[0] - 1))
        
                    # Replace area
                    heatmap[img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]
        
                Heatmaps.append(torch.from_numpy(heatmap).to(torch.float32))
        else:
            Heatmaps = [torch.zeros((heatmap_hw[i][0], heatmap_hw[i][1]), dtype=torch.float32) for i in range(len(heatmap_hw))]
        
        return Heatmaps


    def gaussian_kernel(self, kernel_radius, sigma=2):
        # generate gaussian kernel(not normalized)
        kernel_size = (2 * kernel_radius[1] + 1, 2 * kernel_radius[0] + 1)
        kernel = np.zeros(kernel_size, dtype=np.float32)
        x_center, y_center = kernel_size[1] // 2, kernel_size[0]//2
        for x in range(kernel_size[1]):
            for y in range(kernel_size[0]):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
        # print(kernel)
        return kernel
