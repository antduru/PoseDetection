import torch
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image, resized_crop
import numpy as np
from PIL import Image
import json
import numpy as np 
import math
import matplotlib.pyplot as plt 
import os
import src.arch.models as models

json_path = './annotations/mpii/fullannotations.json'
joint_path = './annotations/mpii/jointdict.json'
jointtable_path = './annotations/mpii/jointtable.json'
jointtable_tensor_path = './images/mpii_table/jointtable.pt'
histogram_path = './annotations/mpii/histogram.json'
images_path = './images/mpii_resized'
mapp_path = './images/mpii_map'

with open(json_path, 'r') as fp:
    json_list = json.load(fp)

with open(jointtable_path, 'r') as fp:
    joint_table = json.load(fp)

with open(joint_path, 'r') as fp:
    joint_dict = json.load(fp)

joint_table_tensor = torch.load(jointtable_tensor_path)

def visualize_histogram(histogram_path, end = 50, grid = (2, 2)):
    
    def plot(i, histogram, title, color, norm=True):
        histogram = np.array(histogram)
        if norm:
            histogram = histogram / np.sum(histogram)
        
        plt.subplot(*grid, i)
        plt.bar(range(1, end+1), histogram, color=color)
        plt.title(title)
        plt.ylabel('probability')
        plt.xlabel('pixel length')
    
    with open(histogram_path, 'r') as fp:
        joint_histogram = json.load(fp)

    plt.figure(figsize=(20, 10))
    
    plot(1, joint_histogram['u_arm'][:end], 'Upper Arm Length', 'g')
    plot(2, joint_histogram['l_arm'][:end], 'Lower Arm Length', 'g')
    plot(3, joint_histogram['u_leg'][:end], 'Upper Leg Length', 'g')
    plot(4, joint_histogram['l_leg'][:end], 'Lower Leg Length', 'g')
    plt.show()

def presente(index):
    def plot_line(coor1, coor2, marker='o', linewidth=4, color=None):
        if coor1 and coor2: 
            x1, y1 = coor1
            x2, y2 = coor2
            plt.plot([x1, x2], [y1, y2], marker=marker, linewidth=linewidth, color=color)
            
    def plot_rect(x1, y1, x2, y2):
        plot_line((x1, y1), (x1, y2), '', 1, 'r')
        plot_line((x1, y2), (x2, y2), '', 1, 'r')
        plot_line((x2, y2), (x2, y1), '', 1, 'r')
        plot_line((x2, y1), (x1, y1), '', 1, 'r')

    def get_joint(joint_id, joint_list):
        for joint in joint_list:
            if joint['id'] == joint_id: return joint
        return None

    def find(joint_list, lambda_fn):
        _ss = joint_list[0]
        for e in joint_list:
            if not lambda_fn(e, _ss): continue
            _ss = e
        return _ss

    def head_width(head_c, image_shape):
        width = abs(head_c['x1'] - head_c['x2']) * image_shape['width']
        height = abs(head_c['y1'] - head_c['y2']) * image_shape['height']
        return (width + height) // 4

    def draw_person_rect(joint_list, head_coor, image_shape):
        c = head_width(head_coor, image_shape)
        
        minx = find(joint_list, lambda c, e: c['x'] < e['x'])['x'] * image_shape['width']
        maxx = find(joint_list, lambda c, e: c['x'] > e['x'])['x'] * image_shape['width']
        miny = find(joint_list, lambda c, e: c['y'] < e['y'])['y'] * image_shape['height']
        maxy = find(joint_list, lambda c, e: c['y'] > e['y'])['y'] * image_shape['height']
        
    #     print(minx, miny, maxx, maxy)
        
        plot_rect(max(minx - c, 0), 
                max(miny - c, 0), 
                min(maxx + c, image_shape['width']), 
                min(maxy + 2*c, image_shape['height']))
                
            
    def present_body_parts(image_array, image_object, joint_dict, title, draw_head=True, draw_bbox=False, draw_pose=True, xlim=None, draw_vector=False):
        plt.figure(figsize=(10, 20))
        plt.axis('off')
        if xlim:
            plt.xlim(*xlim)
        plt.imshow(image_array)
        width = image_object['image_shape']['width']
        height = image_object['image_shape']['height']

        for person in image_object['people']:
            joint_list = person['joints']
            joint_map = {}
            for k in joint_dict:
                joint_temp = get_joint(joint_dict[k], joint_list)
                if not joint_temp: 
                    joint_map[k] = None
                else:
                    joint_map[k] = (joint_temp['x'] * width, joint_temp['y'] * height)
            if draw_pose:
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
            
    #         if not 'head_coordinates' in person: continue
                
            person_head = person['head_coordinates']
            x1 = person_head['x1'] * width
            y1 = person_head['y1'] * height
            x2 = person_head['x2'] * width
            y2 = person_head['y2'] * height
            
            if draw_head:
                plot_rect(x1, y1, x2, y2) # draw head
            
            if draw_bbox:
                draw_person_rect(joint_list, person_head, image_object['image_shape']) # draw body
            
            if len(image_object['people']) >= 2 and draw_vector:
                person_boxes1 = image_object['people'][0]['bbox']
                person_boxes2 = image_object['people'][1]['bbox']
                
                x1 = (person_boxes1['x1'] + person_boxes1['x2']) / 2
                y1 = (person_boxes1['y1'] + person_boxes1['y2']) / 2
                x2 = (person_boxes2['x1'] + person_boxes2['x2']) / 2
                y2 = (person_boxes2['y1'] + person_boxes2['y2']) / 2
                
                args = (x1*width, y1*height, (x2-x1)*width, (y2-y1)*height)
                
                plt.arrow(*args, width=8, color='g')
                # print(args)
        
        plt.show()
        
    def show_image(image_object, images_path):
        image_name = image_object['image_name']
        image_array = np.array(Image.open(os.path.join(images_path, image_name)))
        
        present_body_parts(image_array, 
                            image_object, 
                            joint_dict, 
                            f'Annotation of {image_name}',
                            draw_head=True, 
                            draw_bbox=True, 
                            draw_pose=True, 
                            xlim=None, 
                            draw_vector=False)

        present_body_parts(image_array * 0, 
                            image_object, 
                            joint_dict, 
                            f'Annotation of {image_name}',
                            draw_head=True, 
                            draw_bbox=True, 
                            draw_pose=True, 
                            xlim=None, 
                            draw_vector=False)

    show_image(json_list[index], images_path)

def what_network_sees(index, personindex):
    def what_network_sees2(image_object, images_path, person_index=0, IN_SHAPE=(226, 226)):
        image_name = image_object['image_name']
        original_tensor = to_tensor(Image.open(os.path.join(images_path, image_name)))

        width = image_object['image_shape']['width']
        height = image_object['image_shape']['height']

        if len(image_object['people']) < person_index: return False

        bbox = image_object['people'][person_index]['bbox']
        
        top = int(bbox['y1'] * height)
        left = int(bbox['x1'] * width)
        height = int((bbox['y2'] - bbox['y1']) * height)
        width = int((bbox['x2'] - bbox['x1']) * width)
        cropped_tensor = resized_crop(original_tensor, top, left, height, width, IN_SHAPE)
        # cropped_tensor = crop.resize(1, 3, *IN_SHAPE)

        # return to_pil_image(original_tensor), to_pil_image(cropped_tensor)
        return to_pil_image(cropped_tensor)
    return what_network_sees2(json_list[index], images_path, person_index=personindex)

def image_estimation_list_to_people_list(image_estimation_list, index=5):
    def output_table_to_joint_list(output_table, index):
        joint_list = []
        for k in joint_dict:
            joint_id = joint_dict[k]
            x, y, v, m, dx, dy = tuple(output_table[index, joint_id])
            if m <= 0.5: continue
            joint_list.append({
                'x': float(x),
                'y': float(y),
                'dx': float(dx), # these are additional
                'dy': float(dy), # these are additional
                'id': joint_id,
                'is_visible': bool(v > 0.5)
            })
        return joint_list
    people_list = []
    for (output_table, out_bbox) in image_estimation_list:
        people_list.append({
            'bbox': out_bbox,
            'joints': output_table_to_joint_list(output_table, index)
        })
    return people_list

def plot_line(coor1, coor2, marker='o', linewidth=4, color=None):
    if coor1 and coor2: 
        x1, y1 = coor1
        x2, y2 = coor2
        plt.plot([x1, x2], [y1, y2], marker=marker, linewidth=linewidth, color=color)

def plot_rect(x1, y1, x2, y2):
    plot_line((x1, y1), (x1, y2), '', 1, 'r')
    plot_line((x1, y2), (x2, y2), '', 1, 'r')
    plot_line((x2, y2), (x2, y1), '', 1, 'r')
    plot_line((x2, y1), (x1, y1), '', 1, 'r')

def get_joint(joint_id, joint_list):
    for joint in joint_list:
        if joint['id'] == joint_id: return joint
    return None

def find(joint_list, lambda_fn):
    _ss = joint_list[0]
    for e in joint_list:
        if not lambda_fn(e, _ss): continue
        _ss = e
    return _ss

def head_width(head_c, image_shape):
    width = abs(head_c['x1'] - head_c['x2']) * image_shape['width']
    height = abs(head_c['y1'] - head_c['y2']) * image_shape['height']
    return (width + height) // 4

def draw_person_rect(joint_list, head_coor, image_shape):
    c = head_width(head_coor, image_shape)

    minx = find(joint_list, lambda c, e: c['x'] < e['x'])['x'] * image_shape['width']
    maxx = find(joint_list, lambda c, e: c['x'] > e['x'])['x'] * image_shape['width']
    miny = find(joint_list, lambda c, e: c['y'] < e['y'])['y'] * image_shape['height']
    maxy = find(joint_list, lambda c, e: c['y'] > e['y'])['y'] * image_shape['height']

    plot_rect(max(minx - c, 0), 
            max(miny - c, 0), 
            min(maxx + c, image_shape['width']), 
            min(maxy + 2*c, image_shape['height']))

def visualize(image_tensor, people_list, draw_bbox=True):
    image_array = np.array(to_pil_image(image_tensor))
    c, height, width = image_tensor.shape
    
    plt.figure(figsize=(10, 20))
    plt.axis('off')
    plt.imshow(image_array)
    
    for person in people_list:
        joint_list = person['joints']
        joint_map = {}
        for k in joint_dict:
            joint_temp = get_joint(joint_dict[k], joint_list)
            if not joint_temp: 
                joint_map[k] = None
            else:
                joint_map[k] = (joint_temp['x'] * width, joint_temp['y'] * height)

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
        
        if draw_bbox:
            bbox = person['bbox']
            plot_rect(bbox['x1']*width, bbox['y1']*height, bbox['x2']*width, bbox['y2']*height)
                
    plt.show()

def test_visualize_ds(visualize, image_index=0):
    
    image_object = json_list[image_index]
    image_path = os.path.join(images_path, image_object['image_name'])
    
    input_tensor = to_tensor(Image.open(image_path))
    people_list = image_object['people']
    
    return visualize(input_tensor, people_list)

def test_visualize_model(visualize, model, image_index=0, sppe_index=5):
    
    image_object = json_list[image_index]
    image_path = os.path.join(images_path, image_object['image_name'])
    input_tensor = to_tensor(Image.open(image_path))
        
    yolo_people_list = image_object['people'] # normally use yolo but now just gid
        
    image_estimation_list = models.estimate_for_all(input_tensor, model, yolo_people_list)
    people_list = image_estimation_list_to_people_list(image_estimation_list, index=sppe_index)
    
    return visualize(input_tensor, people_list)

# BIGBOI STUFF NOW
'''
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
import loss


def translate_output_to_joint_list(output_table, joint_dict, p_index):
    joints = []
    for k in joint_dict:
        joint_id = joint_dict[k]
        x, y, v, m, dx, dy = tuple(output_table[p_index, joint_id])

        if m < 0.5: continue

        joint_object = {
            'x': x,
            'y': y,
            'dx': dx,
            'dy': dy,
            'id': joint_id,
            'is_visible': v > 0.5
        }
        joints.append(joint_object)
    
    return joints

def generate():
    with open('./annotations/mpii/fullannotations.json') as fp:
        annotation_list = json.load(fp)

    with open('./annotations/mpii/jointtable.json') as fp:
        jointtable_list = json.load(fp)

    joint_tensor = torch.load('./images/mpii_table/jointtable.pt')

    criterion = loss.CustomLoss()
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

                image_object = generate_image_object(image_object, output_table, image_object)

                present_body_parts(image_array, 
                        image_object, 
                        joint_dict, 
                        f'Annotation of {image_name}',
                        draw_head=True, 
                        draw_bbox=True, 
                        draw_pose=True, 
                        xlim=None, 
                        draw_vector=True)
            

            loss = criterion(output_table, truth_table)
            print(float(loss))

generate()
'''