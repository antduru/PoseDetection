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
OUT_CLASS = 9

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
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, (OUT_CLASS * 16 * 6))
        ])
        self.sigmoid1 = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()

    def forward(self, original_tensor, bboxes, bbox_index):
        out0 = resized_crop(original_tensor, bboxes[bbox_index])
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.linear1(out5)
        out7 = torch.reshape(out6, (OUT_CLASS, 16, 6))
        helper1 = self.relu1(out7[:, :, (0, 1)])     # activate x and y
        helper2 = self.tanh1(out7[:, :, (2, 3)])     # activate dx and dy
        helper3 = self.sigmoid1(out7[:, :, (4, 5)])  # activate c and m
        out8 = torch.cat([helper1, helper2, helper3], dim=2)
        return out8

'''
    The output of the model is 9x16x6

    9 means the number of estimations, current, up, down, left, right and diagonals
    16 means the joints,
    6 means; x, y coordinates, xd, yd direction vectors, c confidence, and m binary mask.
        x, y: coordinates of the joint
        xd, yd: direction vector of the joint. For example, knees point to the hips. (See the pointing diagram)
        c: confidence value
        m: binary mask, if network is not sure this is detected, then fires this neuron.
'''