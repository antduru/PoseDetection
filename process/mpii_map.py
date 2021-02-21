import torch
import torchvision
from scipy.io import loadmat
import numpy as np
from PIL import Image
import json
import cv2
import numpy as np 
import os 
from tqdm import tqdm

def get_joint(joint_id, person_object):
    for joint in person_object['joints']:
        if joint['id'] == joint_id: return joint
    return None

def main():
    image_path = './images/mpii_resized'
    image_map_path = './images/mpii_map'
    json_path = './annotations/mpii/fullannotations.json'
    joint_path = './annotations/mpii/jointdict.json'

    with open(json_path, 'r') as fp:
        json_list = json.load(fp)

    with open(joint_path, 'r') as fp:
        joint_dict = json.load(fp)

    
    for i, json_object in (enumerate(json_list)):
        image_name = json_object['image_name']
        if json_object['image_name'] == '098312641.jpg':
            image_filename = os.path.join(image_path, image_name)

            # print(json_object['type'])
            # image_tensor = np.array(Image.open(image_filename))
            # c, w, h = image_tensor.shape

            # print(len(json_object['people']))

            # body_map = np.zeros((w, h, 1), dtype=np.int32) # body map
            # head_map = np.zeros((w, h, 1), dtype=np.int32) # head map
            # uarm_map = np.zeros((w, h, 1), dtype=np.int32) # upper arm map
            # larm_map = np.zeros((w, h, 1), dtype=np.int32) # lower arm map
            # uleg_map = np.zeros((w, h, 1), dtype=np.int32) # upper leg map
            # lleg_map = np.zeros((w, h, 1), dtype=np.int32) # lower leg map

            for person_object in json_object['people']:
                l_shoulder = get_joint(joint_dict['l_shoulder'], person_object)
                r_shoulder = get_joint(joint_dict['r_shoulder'], person_object)
                pelvis = get_joint(joint_dict['pelvis'], person_object)

                print(f"->{person_object['type']}")

                # for joint in person_object['joints']:
                #     print(joint) 

if __name__ == '__main__': main()