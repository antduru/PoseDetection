import torch
import torchvision
import numpy as np
from PIL import Image
import json
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import os 

def get_joint(joint_id, joint_list):
    for joint in joint_list:
        if joint['id'] == joint_id: return joint
    return None

def plot_line(coor1, coor2):
    if coor1 and coor2: 
        x1, y1 = coor1
        x2, y2 = coor2
        plt.plot([x1, x2], [y1, y2], marker = 'o')

def present_pose(image_array, people_list, joint_dict):
    plt.figure(figsize=(10, 20))
    plt.imshow(image_array)

    for person in people_list:
        joint_list = person['joints']
        joint_map = {}
        for k in joint_dict:
            joint_temp = get_joint(joint_dict[k], joint_list)
            if not joint_temp: 
                joint_map[k] = None
            else:
                joint_map[k] = (joint_temp['x'], joint_temp['y'])

        # legs
        plot_line(joint_map['r_ankle'], joint_map['r_knee'])
        plot_line(joint_map['r_knee'], joint_map['r_hip'])
        plot_line(joint_map['l_ankle'], joint_map['l_knee'])
        plot_line(joint_map['l_knee'], joint_map['l_hip'])

        # arms
        plot_line(joint_map['r_wrist'], joint_map['r_elbow'])
        plot_line(joint_map['r_elbow'], joint_map['r_shoulder'])
        plot_line(joint_map['l_wrist'], joint_map['l_elbow'])
        plot_line(joint_map['l_elbow'], joint_map['l_shoulder'])

        # head
        plot_line(joint_map['upper_neck'], joint_map['head_top'])
        plot_line(joint_map['upper_neck'], joint_map['thorax'])

        # body
        plot_line(joint_map['r_shoulder'], joint_map['thorax'])
        plot_line(joint_map['l_shoulder'], joint_map['thorax'])

        plot_line(joint_map['r_hip'], joint_map['pelvis'])
        plot_line(joint_map['l_hip'], joint_map['pelvis'])

        plot_line(joint_map['thorax'], joint_map['pelvis'])
    
    plt.show()


def main():
    json_path = './annotations/mpii/fullannotations.json'
    joint_path = './annotations/mpii/jointdict.json'
    images_path = './images/mpii_resized'
    mapp_path = './images/mpii_map'

    with open(json_path, 'r') as fp:
        json_list = json.load(fp)

    with open(joint_path, 'r') as fp:
        joint_dict = json.load(fp)

    image_object = json_list[53]
    print(image_object['image_name'])
    image_file = os.path.join(images_path, image_object['image_name'])
    image_array = np.array(Image.open(image_file))
    present_pose(image_array, image_object['people'], joint_dict)

main()