from shutil import copyfile
import json 
import os 
from tqdm import tqdm
from PIL import Image
import src.util as util
from shutil import copyfile

'''
    This calculates the joint table for each person in the dataset. The output of this is
    used on the loss function.
'''

json_path = './annotations/mpii/fullannotations.json'
joint_table_path = './annotations/mpii/jointtable.json'
IOU = 0.5

def generate_table():
    

with open(json_path, 'r') as fp:
    json_file = json.load(fp)

joint_table_list = []

for image_object in json_file:
    image_name = image_object['image_name']

    for i, current_person_object in enumerate(image_object['people']):

        for j, neigbor_person_object in enumerate(image_object['people']):
            c_bbox = current_person_object['bbox']
            n_bbox = neigbor_person_object['bbox']

            if i == j: continue # pass if we are looking at the same person
            if util.iou(c_bbox, n_bbox) < IOU: continue # pass if they are not overlapping

            diagonal_id = util.diagonal_id(c_bbox, n_bbox)

    joint_table_list.append()