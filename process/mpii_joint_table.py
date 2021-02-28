from shutil import copyfile
import json 
import os 
from tqdm import tqdm
from PIL import Image
import util as util
from shutil import copyfile
import torch

'''
    This calculates the joint table for each person in the dataset. The output of this is
    used on the loss function.
'''

json_path = './annotations/mpii/fullannotations.json'
joint_dict_path = './annotations/mpii/jointdict.json'
joint_table_path = './annotations/mpii/jointtable.json'
joint_table_tensor_path = './images/mpii_table/jointtable.pt'
IOU = 0.5

with open(json_path, 'r') as fp:
    annotation_list = json.load(fp)

with open(joint_dict_path, 'r') as fp:
    joint_dict = json.load(fp)

def generate_affinity_vector(current_person_object, k, joint_dict):
    if 'r_ankle' in k: return util.dist(current_person_object, k, 'r_knee', joint_dict)
    elif 'r_knee' in k: return util.dist(current_person_object, k, 'r_hip', joint_dict)
    elif 'r_hip' in k: return util.dist(current_person_object, k, 'pelvis', joint_dict)
    elif 'l_hip' in k: return util.dist(current_person_object, k, 'pelvis', joint_dict)
    elif 'l_knee' in k: return util.dist(current_person_object, k, 'l_hip', joint_dict)
    elif 'l_ankle' in k: return util.dist(current_person_object, k, 'l_knee', joint_dict)
    elif 'pelvis' in k: return util.dist(current_person_object, k, 'thorax', joint_dict)
    elif 'thorax' in k: return util.dist(current_person_object, k, 'pelvis', joint_dict)
    elif 'upper_neck' in k: return util.dist(current_person_object, k, 'thorax', joint_dict)
    elif 'head_top' in k: return util.dist(current_person_object, k, 'upper_neck', joint_dict)
    elif 'r_wrist' in k: return util.dist(current_person_object, k, 'r_elbow', joint_dict)
    elif 'r_elbow' in k: return util.dist(current_person_object, k, 'r_shoulder', joint_dict)
    elif 'r_shoulder' in k: return util.dist(current_person_object, k, 'thorax', joint_dict)
    elif 'l_shoulder' in k: return util.dist(current_person_object, k, 'thorax', joint_dict)
    elif 'l_elbow' in k: return util.dist(current_person_object, k, 'l_shoulder', joint_dict)
    elif 'l_wrist' in k: return util.dist(current_person_object, k, 'l_elbow', joint_dict)
    return 0, 0

def generate_joint_table(current_person_object, joint_dict):
    # (x, y, dx, dy, v, m)
    # x, y: coordinates of the joint (absolute coordinates)
    # dx, dy: affinity vector
    # v: is visible
    # c: confidence, 1 sure, 0 not sure, loss is according to it!

    # joint_table = [[[0, 0, 0, 0, 0, 0] for _ in range(16)] for _ in range(9)] do not store this on json!
    joint_table = torch.zeros((1, 9, 16, 6), dtype=torch.float)

    for k in joint_dict:
        joint = util.get_joint(joint_dict[k], current_person_object['joints'])

        if joint:
            joint_id = joint['id']
            pos_x = joint['x']
            pos_y = joint['y']
            v = 1 if joint['is_visible'] else 0

            d_x, d_y = generate_affinity_vector(current_person_object, k, joint_dict)

            joint_table[0, 5, joint_id, 0] = pos_x
            joint_table[0, 5, joint_id, 1] = pos_y
            joint_table[0, 5, joint_id, 2] = d_x
            joint_table[0, 5, joint_id, 3] = d_y
            joint_table[0, 5, joint_id, 4] = v
            joint_table[0, 5, joint_id, 5] = 1

    return joint_table

def mutate_joint_table(joint_table, current_person, neigbor_person, diagonal_id, joint_dict):
    for k in joint_dict:
        joint = util.get_joint(joint_dict[k], neigbor_person['joints'])

        if not joint: continue
        if not util.is_joint_inside(current_person, joint): continue

        joint_id = joint['id']
        pos_x = joint['x']
        pos_y = joint['y']
        v = 1 if joint['is_visible'] else 0

        d_x, d_y = generate_affinity_vector(neigbor_person, k, joint_dict)

        joint_table[0, diagonal_id - 1, joint_id, 0] = pos_x
        joint_table[0, diagonal_id - 1, joint_id, 1] = pos_y
        joint_table[0, diagonal_id - 1, joint_id, 2] = d_x
        joint_table[0, diagonal_id - 1, joint_id, 3] = d_y
        joint_table[0, diagonal_id - 1, joint_id, 4] = v
        joint_table[0, diagonal_id - 1, joint_id, 5] = 1

    return joint_table 

joint_table_list = []
joint_table_tensor_array = []

for image_object in tqdm(annotation_list):
    people_joint_list = []

    for i, current_person_object in enumerate(image_object['people']):
        joint_table = generate_joint_table(current_person_object, joint_dict)
        num_occlusions = 0

        for j, neigbor_person_object in enumerate(image_object['people']):
            c_bbox = current_person_object['bbox']
            n_bbox = neigbor_person_object['bbox']

            if i == j: continue # pass if we are looking at the same person
            if util.iou(c_bbox, n_bbox) < IOU: continue # pass if they are not overlapping

            diagonal_id = util.diagonal_id(c_bbox, n_bbox)

            num_occlusions += 1
            joint_table = mutate_joint_table(
                joint_table, 
                current_person_object, 
                neigbor_person_object, 
                diagonal_id, 
                joint_dict)
        
        joint_table_tensor_array.append(joint_table)
        people_joint_list.append({
            'i': len(joint_table_tensor_array) - 1,
            'num_occlusions': num_occlusions
        }) # add joint table for ith person
    joint_table_list.append({
        'image_name': image_object['image_name'],
        'people': people_joint_list
    })
    # break

print('Done! Saving Joint Table and Generated JSON file...')

with open(joint_table_path, 'w+') as fp:
    json.dump(joint_table_list, fp)

torch.save(torch.cat(joint_table_tensor_array), joint_table_tensor_path)

print('Processing finished!')
print("Reconstructed {} people's joint table.".format(len(joint_table_tensor_array)))