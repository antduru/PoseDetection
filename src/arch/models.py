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
import math

IN_SHAPE = (226, 226)
OUT_SHAPE = (9, 16, 6)

class CustomVgg19(nn.Module):
    def __init__(self, k=12):
        ''' get first k including k's layer -> 0 to 36 '''
        super().__init__()
        original = torchvision.models.vgg19(pretrained=True)
        features = []

        for i, module in enumerate(next(original.children())):
            features.append(module)

        self.features = nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

# def resized_crop(original_tensor, bbox):
#     top = bbox['y1']
#     left = bbox['x1']
#     height = bbox['y2'] - bbox['y1']
#     width = bbox['x2'] - bbox['x1']
#     crop = F.resized_crop(original_tensor, top, left, height, width, SIZE)
#     return crop.resize(1, 3, *SIZE)

def resized_crop(original_tensor, bbox):
    if bbox:
        c, height, width = original_tensor.shape
        
        top = int(bbox['y1'] * height)
        left = int(bbox['x1'] * width)
        height = int((bbox['y2'] - bbox['y1']) * height)
        width = int((bbox['x2'] - bbox['x1']) * width)
    else:
        top = 0
        left = 0
        height = 100
        width = 100

    cropped_tensor = F.resized_crop(original_tensor, top, left, height, width, IN_SHAPE)
    return cropped_tensor

class SinglePersonPoseEtimator(nn.Module):
    def __init__(self):
        ''' Architecture is similar to Yolov1 '''
        super().__init__()

        self.vgg =  CustomVgg19()

        # freeze the vgg network
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, math.prod(OUT_SHAPE))
        )
        self.sigmoid1 = nn.Sigmoid()
        self.tanh1 = nn.Tanh()

    def forward(self, original_tensor, bbox):
        cropped_tensor = resized_crop(original_tensor, bbox)

        if cropped_tensor.ndim != 4:
            cropped_tensor = torch.unsqueeze(cropped_tensor, 0)

        out0 =  self.vgg(cropped_tensor)
        out1 = self.conv1(out0)
        out2 = self.linear1(out1)

        out3 = torch.reshape(out2, OUT_SHAPE)

        helper1 = self.sigmoid1(out3[:, :, :4])     # activate x, y, v, m
        helper2 = self.tanh1(out3[:, :, 4:])              # activate dx and dy

        out3 = torch.cat([helper1, helper2], dim=2)

        return out3
'''
    The output of the model is 9x16x6

    9 means the number of estimations, current, up, down, left, right and diagonals
    16 means the joints,
    6 means; x, y coordinates, xd, yd direction vectors, c confidence, and m binary mask.
        x, y: coordinates of the joint
        v: visible value
        m: binary mask, if network is not sure this is detected, then fires this neuron.
        xd, yd: direction vector of the joint. For example, knees point to the hips. (See the pointing diagram)
'''

if __name__ == "__main__":
    model = SinglePersonPoseEtimator()
    model.eval()

    with torch.no_grad():
        resized_tensor = torch.zeros((1, 3, 226, 226))
        out = model(resized_tensor, None)

        print(out)
