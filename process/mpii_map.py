import torch
import torchvision
import numpy as np
from PIL import Image
import json
import numpy as np 

def get_joint(joint_id, joint_list):
    for joint in joint_list:
        if joint['id'] = joint_id: return joint
    return None


if __name__ == '__main__':
    json_path = './annotations/mpii/fullannotations.json'
    joint_path = './annotations/mpii/jointdict.json'
    images_path = './images/mpii_resized'
    mapp_path = './images/mpii_map'

    joint_histogram = {
        'u_arm': np.zeros((1000,)),
        'l_arm': np.zeros((1000,)),
        'u_leg': np.zeros((1000,)),
        'l_leg': np.zeros((1000,)),
        'body': np.zeros((1000,)),
        'head': np.zeros((1000,)),
    }

    with open(json_path, 'r') as fp:
        json_list = json.load(fp)

    with open(joint_path, 'r') as fp:
        joint_dict = json.load(fp)

    
    for image_object in json_list:
        image_name = image_object['image_name']

        for person_object in image_object['people']:


    

    
