import torch
import torchvision
import numpy as np
from PIL import Image
import json
import numpy as np 
import math

def get_joint(joint_id, joint_list):
    for joint in joint_list:
        if joint['id'] == joint_id: return joint
    return None

def distance(j1, j2):
    return int(math.sqrt((j1['x'] - j2['x'])**2 + (j1['y'] - j2['y'])**2))

def add_histogram(jistogram_list, joint1, joint2):
    if not joint1 or not joint2: return False
    if not joint1['is_visible'] or not joint2['is_visible']: return False

    int_dist = distance(joint1, joint2)
    jistogram_list[int_dist] += 1
    return True


if __name__ == '__main__':
    json_path = './annotations/mpii/fullannotations.json'
    joint_path = './annotations/mpii/jointdict.json'
    histogram_path = './annotations/mpii/histogram.json'
    images_path = './images/mpii_resized'
    mapp_path = './images/mpii_map'

    joint_histogram = {
        'u_arm': np.zeros((1000,), dtype=np.int32),
        'l_arm': np.zeros((1000,), dtype=np.int32),
        'u_leg': np.zeros((1000,), dtype=np.int32),
        'l_leg': np.zeros((1000,), dtype=np.int32),
        'body': np.zeros((1000,), dtype=np.int32),
        'head': np.zeros((1000,), dtype=np.int32),
    }

    with open(json_path, 'r') as fp:
        json_list = json.load(fp)

    with open(joint_path, 'r') as fp:
        joint_dict = json.load(fp)

    
    for image_object in json_list:
        image_name = image_object['image_name']

        for person_object in image_object['people']:
            r_elbow = get_joint(joint_dict['r_elbow'], person_object['joints'])
            r_wrist = get_joint(joint_dict['r_wrist'], person_object['joints'])
            r_shoulder = get_joint(joint_dict['r_shoulder'], person_object['joints'])
            r_ankle = get_joint(joint_dict['r_ankle'], person_object['joints'])
            r_knee = get_joint(joint_dict['r_knee'], person_object['joints'])
            r_hip = get_joint(joint_dict['r_hip'], person_object['joints'])

            l_elbow = get_joint(joint_dict['l_elbow'], person_object['joints'])
            l_wrist = get_joint(joint_dict['l_wrist'], person_object['joints'])
            l_shoulder = get_joint(joint_dict['l_shoulder'], person_object['joints'])
            l_ankle = get_joint(joint_dict['l_ankle'], person_object['joints'])
            l_knee = get_joint(joint_dict['l_knee'], person_object['joints'])
            l_hip = get_joint(joint_dict['l_hip'], person_object['joints'])

            head_top = get_joint(joint_dict['head_top'], person_object['joints'])
            upper_neck = get_joint(joint_dict['upper_neck'], person_object['joints'])

            # arms
            add_histogram(joint_histogram['l_arm'], r_elbow, r_wrist)
            add_histogram(joint_histogram['u_arm'], r_elbow, r_shoulder)
            add_histogram(joint_histogram['l_arm'], l_elbow, l_wrist)
            add_histogram(joint_histogram['u_arm'], l_elbow, l_shoulder)

            # legs
            add_histogram(joint_histogram['l_leg'], r_knee, r_ankle)
            add_histogram(joint_histogram['u_leg'], r_knee, r_hip)
            add_histogram(joint_histogram['l_leg'], l_knee, l_ankle)
            add_histogram(joint_histogram['u_leg'], l_knee, l_hip)

            # head
            add_histogram(joint_histogram['head'], head_top, upper_neck)

            # body
            # havent added yet!
            
    
    for k in joint_histogram:
        joint_histogram[k] = [int(e) for e in joint_histogram[k]]
    
    with open(histogram_path, 'w+') as fp:
        json.dump(joint_histogram, fp, indent=1)

                