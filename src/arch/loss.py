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

class MyDotOperation():
    def __call__(self, Yhat, Y):
        return torch.sum(Yhat * Y, dim=2, keepdim=True)

class CustomLoss():
    def __init__(self):
        self.cross = MyCrossEntropyOperation()
        self.dot = MyDotOperation()
        self.means = nn.MSELoss()
    
    def __call__(self, output_table, truth_table):

        out_dirr = output_table[:, :, (4, 5)].clone()
        out_coor = output_table[:, :, (0, 1)].clone()
        out_viss = output_table[:, :, (2,)].clone()
        out_mask = output_table[:, :, (3,)].clone()

        tru_dirr = truth_table[:, :, (4, 5)].clone()
        tru_coor = truth_table[:, :, (0, 1)].clone()
        tru_viss = truth_table[:, :, (2,)].clone()
        tru_mask = truth_table[:, :, (3,)].clone()

        direction_loss = torch.sum(out_dirr * tru_dirr, dim=2, keepdim=True)
        joint_loss = torch.abs(out_coor - tru_coor)
        visible_loss = self.cross(out_viss, tru_viss)

        dirjovis_loss = torch.sum(direction_loss + joint_loss + visible_loss, dim=2, keepdim=True)

        # 
        mask_loss = self.cross(out_mask, tru_mask)

        # return tru_mask

        full_loss    = torch.where(tru_mask >  0.5, 1., 0.) # truth_table[:, :, (3,)] > 0.5
        m_only_loss  = torch.where(torch.where(tru_mask >  0.5, 0., 1.) * out_mask >  0.5, 1., 0.) # truth_table[:, :, (3,)] < 0.5 and output_table[:, :, (3,)] > 0.5
        no_loss      = torch.where(torch.where(tru_mask >  0.5, 0., 1.) * out_mask <= 0.5, 1., 0.) # truth_table[:, :, (3,)] < 0.5 and output_table[:, :, (3,)] < 0.5

        total_loss = (dirjovis_loss * m_only_loss + mask_loss) * no_loss
        '''
                out m | truth m
                  1        1    -> apply loss to all
                  1        0    -> apply loss to all
                  0        1    -> apply loss only to m
                  0        0    -> apply no loss
        '''
        return torch.sum(total_loss)

if __name__ == "__main__":
    output_table = torch.random((9, 16, 6)) * 0.5
    truth_table  = torch.random((9, 16, 6)) * 0.5

    loss = CustomLoss()

    out = loss(output_table, truth_table)

    print(out)
