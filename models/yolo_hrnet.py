# Author: Jiahao Li
# CreatTime: 2022/12/2
# FileName: 
# Description: None

import torch
import torch.nn as nn
import torchvision
import time
from models.yolo import DetectionModel
from models.hrnet_module import HighResolutionNet



class yolo_hrnet(DetectionModel):

    def __init__(self,
                # yolo Detcteion Head
                cfg='yolov5s.yaml',
                ch=3,
                nc=None,
                anchors=None,
                # hrnet KeyPoint Head
                hrnet_head=None):

        super(yolo_hrnet, self).__init__(cfg,
                                         ch,
                                         nc,
                                         anchors)

        if hrnet_head is None:
            self.hrnet_head = HighResolutionNet(self.out_channels[self.save[-3:]],
                                                base_channel=32,
                                                num_joints=1)


