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

'''
    This module contains a custom loss function
'''
class MyCrossEntropyOperation():
    def __call__(self, Yhat, Y):
        return - Y * torch.log(Yhat) + (1 - Y) * torch.log(1 - Yhat)

class CustomLoss():
    def __init__(self):
        self.cross = MyCrossEntropyOperation()
        self.means = nn.MSELoss()
    
    def __call__(self, output_table, truth_table, bbox):
        truth_table = truth_table.clone()
        truth_table[:, :, 0] = (bbox['x2'] - bbox['x1']) * (truth_table[:, :, 0] - bbox['x1']) / 256 # convert this to relative coordinates
        truth_table[:, :, 1] = (bbox['y2'] - bbox['y1']) * (truth_table[:, :, 1] - bbox['y1']) / 256

        output_table = output_table[5].reshape(-1, 6)
        truth_table = truth_table[5].reshape(-1, 6)

        true_mask = truth_table[:, 5]
        out_mask = output_table[:, 5]

        coordinates_error = self.means(output_table[:, :4], truth_table[:, :4])
        visible_error = self.cross(output_table[:, 4], truth_table[:, 4])
        mask_error = self.cross(out_mask, true_mask)

        temp = mask_error + (coordinates_error + visible_error) * true_mask
        return torch.sum(temp)



