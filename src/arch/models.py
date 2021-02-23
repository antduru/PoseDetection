import torch
import torch.nn as nn 
import torchvision
from collections import OrderedDict
from torchvision.transforms.transforms import F
from PIL import Image
import copy
import numpy as np 
import os 
import json

SIZE = (256, 256)

def resized_crop(original_tensor, bbox):
    top = bbox['y1']
    left = bbox['x1']
    height = bbox['y2'] - bbox['y1']
    width = bbox['x2'] - bbox['x1']
    crop = F.resized_crop(original_tensor, top, left, height, width, SIZE)
    return crop.resize(1, 3, *SIZE)

class SinglePersonPoseEtimator(nn.Module):
    def __init__(self):
        ''' Architecture is similar to Yolov1 '''
        super().__init__()
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(3, 64, 7, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(64, 200, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(200, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(512, 256, 1, 1, 0), # 1
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, 1, 0), # 2
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, 1, 0), # 3
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, 1, 0), # 4
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.conv5 = nn.Sequential(*[
            nn.Conv2d(1024, 512, 1, 1, 0), # 1
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0), # 2
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 1, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])
        self.linear1 = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(2048, 3072),
            nn.Tanh(),
            nn.Linear(3072, 3072)
        ])

    def forward(self, original_tensor, bboxes, bbox_index):
        out0 = resized_crop(original_tensor, bboxes[bbox_index])
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.linear1(out5)
        out7 = torch.reshape(out6, (32, 16, 6))
        return out7