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
# import loss

# do not change ! iof you do change the loss module also
IN_SHAPE = (226, 226)
OUT_SHAPE = (9, 16, 6)

'''
    This module contains a custom loss function
'''
class MyCrossEntropyOperation():
    def __call__(self, Yhat, Y, epsilon=0.0001):
        return - Y * torch.log(Yhat + epsilon) + (1 - Y) * torch.log(1 - Yhat + epsilon)

class CustomLoss():
    def __init__(self):
        self.cross = MyCrossEntropyOperation()
        self.means = nn.MSELoss()
    
    def __call__(self, output_table, truth_table):

        # for now only consider joint loss
        # print(output_table.shape)
        out_dirr = output_table[5:6, :, 4:6]  # direction vector
        out_coor = output_table[5:6, :, 0:2]  # coordinate vector
        out_viss = output_table[5:6, :, 2:3]    # visible vector
        out_mask = output_table[5:6, :, 3:4]    # mask vector

        tru_dirr = truth_table[5:6, :, 4:6] 
        tru_coor = truth_table[5:6, :, 0:2] 
        tru_viss = truth_table[5:6, :, 2:3] 
        tru_mask = truth_table[5:6, :, 3:4] 

        direction_loss = 1 - torch.sum(out_dirr * tru_dirr, dim=2, keepdim=True) # same as loss = 1 - dot(Out, True)
        joint_loss = torch.abs(out_coor - tru_coor)
        visible_loss = self.cross(out_viss, tru_viss)

        # print(tru_coor)

        dirjovis_loss = torch.sum(direction_loss + joint_loss + visible_loss, dim=2, keepdim=True)
        # dirjovis_loss = torch.sum(direction_loss, dim=2, keepdim=True)

        # 
        mask_loss = self.cross(out_mask, tru_mask)

        # return tru_mask

        # full_loss    = torch.where(tru_mask >  0.5, 1., 0.) # truth_table[:, :, (3,)] > 0.5
        # m_only_loss  = torch.where(torch.where(tru_mask >  0.5, 0., 1.) * out_mask >  0.5, 1., 0.) # truth_table[:, :, (3,)] < 0.5 and output_table[:, :, (3,)] > 0.5
        no_loss      = torch.where(tru_mask * torch.where(torch.abs(out_mask) > 0.5, 1., 0.) < 0.5, 0., 1.) # truth_table[:, :, (3,)] < 0.5 and output_table[:, :, (3,)] < 0.5

        total_loss = (dirjovis_loss + mask_loss) * no_loss
        '''
                out m | truth m
                  1        1    -> apply loss to all
                  1        0    -> apply loss to all
                  0        1    -> apply loss to all
                  0        0    -> apply no loss
        '''
        return torch.sum(total_loss)

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

        # convert the boi
        _, img_height, img_width = original_tensor.shape
        bbox_width = bbox['x2'] - bbox['x1']
        bbox_height = bbox['y2'] - bbox['y1']

        out_coor_x = bbox_width * out3[:, :, (0,)] / (IN_SHAPE[1] * img_width)
        out_coor_y = bbox_height * out3[:, :, (1,)] / (IN_SHAPE[0] * img_height)

        out = torch.cat([out_coor_x, out_coor_y, out3[:, :, 2:]], dim=2)
        # end conver to boi

        return out, bbox
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
    with open('./annotations/mpii/fullannotations.json') as fp:
        annotation_list = json.load(fp)

    with open('./annotations/mpii/jointtable.json') as fp:
        jointtable_list = json.load(fp)

    joint_tensor = torch.load('./images/mpii_table/jointtable.pt')

    criterion = CustomLoss()
    model = SinglePersonPoseEtimator()
    
    print("annotations and joint table are loaded!")

    image_index = 0
    person_index = 0

    for i, (image_object, jointtable_object) in enumerate(zip(annotation_list, jointtable_list)):
        if i != image_index: continue
        image_shape = image_object

        for j, (person_object, people_object2) in enumerate(zip(image_object['people'], jointtable_object['people'])):
            if j != person_index: continue

            with torch.no_grad():
                model.eval()
                image_path = os.path.join('./images/mpii_resized/', image_object['image_name'])
                image_tensor = F.to_tensor(Image.open(image_path))
                bbox = person_object['bbox']

                truth_table = joint_tensor[j]
                output_table, bbox = model(image_tensor, bbox) # for now let them be same
                

                loss = criterion(output_table, truth_table)
                print(float(loss))
