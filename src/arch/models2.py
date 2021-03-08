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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, activation='relu', pool_size=2):
        super().__init__()

        padding = kernel // 2 # same padding
        stride = 1

        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel, stride, padding)
        self.conv_out = nn.Conv2d(2 * in_channels, out_channels, kernel, stride, padding)
        
        if pool_size >= 2:
            self.pool = nn.MaxPool2d(pool_size)
        else:
            self.pool = nn.Identity()

        if 'relu' in activation:
            self.activation = nn.ReLU()
        elif 'sigmoid' in activation:
            self.activation = nn.Sigmoid()
        elif 'tanh' in activation:
            self.activation = nn.Tanh()
        else:
            raise Exception('Cannot find activation named: ' + str(activation))

    def forward(self, x):
        middle = self.activation(self.conv_in(x))
        stacked = torch.cat([middle, x], dim=1)
        result = self.activation(self.conv_out(stacked))
        return self.pool(result)

class ModelPrimus(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(3,   128, 7, 'tanh', 2) # resulting 16x113x113
        self.block2 = ResBlock(128, 64,  7, 'tanh', 2) # resulting 16x56x56
        self.block3 = ResBlock(64,  32,  5, 'tanh', 2) # resulting 16x28x28
        self.block4 = ResBlock(32,  16,  5, 'tanh', 2) # resulting 16x14x14
        self.block5 = ResBlock(16,  16,  3, 'tanh', 2) # resulting 16x7x7
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class ModelSecundus(nn.Module):
    def __init__(self):
        super().__init__()
        self.block6 = ResBlock(16,  16,  3, 'tanh', 0) # resulting 16x7x7
        self.block7 = ResBlock(16,  16,  3, 'tanh', 0) # resulting 16x7x7
        self.identity = nn.Identity()
        self.activation = nn.Tanh()
        self.block8 = ResBlock(48,  96,  3, 'tanh', 2) # resulting 16x3x3
    
    def forward(self, x):
        a = self.identity(x)
        b = self.block6(x)
        c = self.block7(x)

        result = torch.cat([a, b, c], dim=1)

        activated = self.activation(result)

        return self.block8(activated)

class ModelTertius(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(864, 2*864),
            nn.Tanh(),
            nn.Linear(2*864, 3*864),
            nn.Tanh(),
            nn.Linear(3*864, 3*864),
            nn.Tanh(),
            nn.Linear(3*864, 3*864),
            nn.Tanh(),
            nn.Linear(3*864, 2*864),
            nn.Tanh(),
            nn.Linear(2*864, 96))
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.seq1(x)

        a = self.sigm(x[:, :, :4]) # activate x, y, v, m
        b = self.tanh(x[:, :, 4:]) # activate dx and dy

        result = torch.cat([a, b], dim=2)

        return result


x = torch.zeros((1, 3, 226, 226))
model = nn.Sequential(ModelPrimus(), ModelSecundus(), ModelTertius())

yh = model(x)

print(yh.shape)