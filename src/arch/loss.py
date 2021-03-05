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

# do not change ! iof you do change the models module also
IN_SHAPE = (226, 226)
OUT_SHAPE = (9, 16, 6)



if __name__ == "__main__":
    with open('./annotations/mpii/fullannotations.json') as fp:
        annotation_list = json.load(fp)

    with open('./annotations/mpii/jointtable.json') as fp:
        jointtable_list = json.load(fp)

    joint_tensor = torch.load('./images/mpii_table/jointtable.pt')

    criterion = CustomLoss()
    
    print("annotations and joint table are loaded!")



    image_index = 0
    person_index = 0

    for i, (image_object, jointtable_object) in enumerate(zip(annotation_list, jointtable_list)):
        if i != image_index: continue
        image_shape = image_object

        for j, (person_object, people_object2) in enumerate(zip(image_object['people'], jointtable_object['people'])):
            if j != person_index: continue

            truth_table = joint_tensor[j]
            output_table = truth_table # for now let them be same
            

            loss = criterion(output_table, truth_table)
            print(float(loss))
