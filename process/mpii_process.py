import torch
import torchvision
from scipy.io import loadmat
import numpy as np
from PIL import Image
import json

'''
This generates fully annotated images from the MPII matlab file. JSON architecture is given below;

[
    {
        "image_name": string,
        "is_train": boolean,
        "has_people": boolean,
        "people": 
        [
            {
                "type": integer,
                "head_coordinates": 
                {
                    "x1": integer, 
                    "y1": integer, 
                    "x2": integer, 
                    "y2": integer 
                },
                "joints": 
                [ 
                    {
                        "x": integer, 
                        "y": integer, 
                        "id": integer, 
                        "is_visible": boolean 
                    } 
                ]
            }
        ]
    }
]
'''


def main():
    print('Reading matlab file')

    image_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/images'
    matlab_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    json_path = './annotations/mpii/fullannotations.json'

    matfile = loadmat(matlab_path)['RELEASE']

    t1, t2, t3, t4, t5, t6 = matfile.item()

    def iterate_tuple_12(t1, t2):
        image_array = []
        skipped_images = set()
        appened_images = set()
        
        for row_t1, row_t2 in zip(t1[0], t2[0]):
            row_t1 = row_t1.item()
            st1, st2, _, _ = row_t1
            
            image_name = st1.item()[0][0]
            is_train = bool(row_t2 == 1)
            
            # idk why but their annotations are empty so skip
            if st2.size == 0: 
                skipped_images.add(image_name)
                continue
                
            people_array = []
            
            # iterate over people
            for row_st1 in st2[0]:
                if row_st1 is None: continue
                row_st1_item = row_st1.item()
                
                if len(row_st1_item) == 2 and True: # TYPE 1 Annotation
                    if row_st1_item[0].size == 0: continue
                    
                    scale = float(row_st1_item[0].item())
                    objposx = int(row_st1_item[1].item()[0].item())
                    objposy = int(row_st1_item[1].item()[1].item())
                    
                    people_array.append({
                        'type': 1,
                        'scale': scale,
                        'objposx': objposx,
                        'objposy': objposy
                    })
                    
                elif len(row_st1_item) == 4 and True: # TYPE 2 Annotation
                    x1, y1, x2, y2 = row_st1_item
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    people_array.append({
                        'type': 2,
                        'head_coordinates': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        }
                    })
            
                elif len(row_st1_item) == 7 and True: # TYPE 3 Annotation
                    if row_st1_item[4].size == 0: continue
                    
                    x1, y1, x2, y2 = row_st1_item[:4]
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    joint_object = {
                        'type': 3,
                        'head_coordinates': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'joints': []
                    }
                    
                    for joint in row_st1_item[4].item()[0][0]:
                        if len(joint) == 3:
                            x, y, jid = joint
                            x, y, jid = int(x), int(y), int(jid)
                            visible = False
                            joint_object['joints'].append({
                                'x': x,
                                'y': y,
                                'id': jid,
                                'is_visible': visible
                            })
                        else:
                            x, y, jid, visible = joint
                            x, y, jid = int(x), int(y), int(jid)

                            if joint[3].size == 0:
                                visible = False
                            else:
                                visible = bool(visible == 1)
                            joint_object['joints'].append({
                                'x': x,
                                'y': y,
                                'id': jid,
                                'is_visible': visible
                            })
                    people_array.append(joint_object)
            
                elif len(row_st1_item) == 17 and True: # TYPE 3 Annotation
                    if row_st1_item[4].size == 0: continue
                    
                    x1, y1, x2, y2 = row_st1_item[:4]
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    joint_object = {
                        'type': 3,
                        'head_coordinates': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'joints': []
                    }
                    
                    for joint in row_st1_item[4].item()[0][0]:
                        if len(joint) == 3:
                            x, y, jid = joint
                            x, y, jid = int(x), int(y), int(jid)
                            visible = False
                            
                            joint_object['joints'].append({
                                'x': x,
                                'y': y,
                                'id': jid,
                                'is_visible': visible
                            })
                        else:
                            x, y, jid, visible = joint
                            x, y, jid = int(x), int(y), int(jid)

                            if joint[3].size == 0:
                                visible = False
                            else:
                                visible = bool(visible == 1)
                                
                            joint_object['joints'].append({
                                'x': x,
                                'y': y,
                                'id': jid,
                                'is_visible': visible
                            })
                    people_array.append(joint_object)
                    
                elif len(row_st1_item) == 35 and True: # TYPE 3 Annotation
                    if row_st1_item[4].size == 0: continue
                    
                    x1, y1, x2, y2 = row_st1_item[:4]
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    joint_object = {
                        'type': 3,
                        'head_coordinates': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        },
                        'joints': []
                    }
                    
                    for joint in row_st1_item[4].item()[0][0]:
                        x, y, jid, visible = joint
                        x, y, jid = int(x), int(y), int(jid)
                        
                        if joint[3].size == 0:
                            visible = False
                        else:
                            visible = bool(visible == 1)
                            
                        joint_object['joints'].append({
                            'x': x,
                            'y': y,
                            'id': jid,
                            'is_visible': visible
                        })
                    people_array.append(joint_object)

                image_object = {
                    'image_name': image_name,
                    'people': people_array,
                    'is_train': is_train
                }
                image_array.append(image_object)
                appened_images.add(image_name)
            
        return image_array, (skipped_images, appened_images)
            
    image_array, (skipped_images, appened_images) = iterate_tuple_12(t1, t2)

    recovered = (len(appened_images)) / (len(image_array)) * 100
    print(f'JSON process finished, {int(recovered)}% of the images are translated ({len(skipped_images)} skipped, {len(appened_images)} recovered, {len(skipped_images) + len(appened_images)} total).')
    print('Generating full annotations...')

    json_list = image_array

    def append_annotated_set(json_object, train_list):
        image_name = json_object['image_name']
        is_train = json_object['is_train'] # i guess this should be called 'is_test'
        people = json_object['people']

        # if len(people) == 0: return # this should not be in test set

        for person_anno in people:
            if person_anno['type'] != 3: return 

            scale = 1.25

            person_anno['head_coordinates']['x1'] = person_anno['head_coordinates']['x1'] // scale
            person_anno['head_coordinates']['y1'] = person_anno['head_coordinates']['y1'] // scale
            person_anno['head_coordinates']['x2'] = person_anno['head_coordinates']['x2'] // scale
            person_anno['head_coordinates']['y2'] = person_anno['head_coordinates']['y2'] // scale

            for joint in person_anno['joints']:
                joint['x'] = joint['x'] // scale
                joint['y'] = joint['y'] // scale
        
        json_object['has_people'] = len(people) != 0
        train_list.append(json_object)

    train_list = []

    for json_object in json_list:
        append_annotated_set(json_object, train_list)
    
    print(f'Full annotations are generated. {len(train_list)} generated out of {len(image_array)}')
    print('Saving JSON file ...')

    with open(json_path, 'w+') as fp:
        json.dump(train_list, fp, indent=1)

    print('JSON file saved all proprocessig done!')

if __name__ == '__main__': main()