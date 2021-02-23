import torch
import torchvision
from scipy.io import loadmat
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import os 

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
def find(joint_list, lambda_fn):
    _ss = joint_list[0]
    for e in joint_list:
        if not lambda_fn(e, _ss): continue
        _ss = e
    return _ss

def head_width(head_c):
    width = abs(head_c['x1'] - head_c['x2']) 
    height = abs(head_c['y1'] - head_c['y2'])
    return (width + height) // 4

def get_bbox(person_object, image_object):
    c = head_width(person_object['head_coordinates'])
    
    minx = find(person_object['joints'], lambda c, e: c['x'] < e['x'])['x']
    maxx = find(person_object['joints'], lambda c, e: c['x'] > e['x'])['x']
    miny = find(person_object['joints'], lambda c, e: c['y'] < e['y'])['y']
    maxy = find(person_object['joints'], lambda c, e: c['y'] > e['y'])['y']
    
    return {
        'x1': max(minx - c, 0),
        'y1': max(miny - c, 0),
        'x2': min(maxx + c, image_object['image_shape']['width']),
        'y2': min(maxy + 2*c, image_object['image_shape']['height'])
    }

def get_image_shape_dict(old_image_path, image_name, scale_factor):
    image_path = os.path.join(old_image_path, image_name)

    image_array = np.array(Image.open(image_path))
    return {
        'width': image_array.shape[1] // scale_factor,
        'height': image_array.shape[0] // scale_factor
    }

def has(sett, dictt):
    for sette in sett:
        if not sette in dictt: return False
    return True

def main():
    old_image_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/images'
    rawjson_path = './annotations/mpii/raw_annotations.json'
    processedjson_path = './annotations/mpii/fullannotations.json'
    scale_factor = 2

    print('Reading raw json file')

    with open(rawjson_path) as fp:
        raw_json = json.load(fp)['RELEASE']

    print('Processing raw json file')

    processed_json = []
    person_should_include = {'annopoints', 'x1', 'y1', 'x2', 'y2', 'scale'}

    for anno, is_train in zip(tqdm(raw_json['annolist']), raw_json['img_train']):
        if is_train == 0:
            continue
        else:
            image_object = {
                'image_name': anno['image']['name'],
                'is_train': True,
                "has_people": False,
                'people': []
            }
            image_object_people = []
            
    #         if '076161962.jpg' not in anno['image']['name']: continue
            
            exclude = False
            
            if len(anno['annorect']) == 0: continue

            for person in anno['annorect']:
                if not has(person_should_include, person): 
                    exclude = True
                    break
                
                if isinstance(person['scale'], list):
                    exclude = True
                    break

                person_object = {
                    'head_coordinates': {
                        "x1": int(person['x1'] / scale_factor), 
                        "y1": int(person['y1'] / scale_factor), 
                        "x2": int(person['x2'] / scale_factor), 
                        "y2": int(person['y2'] / scale_factor)
                    },
                    'scale': float(person['scale']),
                    'joints': []
                }

                if len(person['annopoints']) == 0: 
                    exclude = True
                    continue

                if not isinstance(person['annopoints']['point'], list):
                    person_joint_list_raw = [person['annopoints']['point']]
                else:
                    person_joint_list_raw = person['annopoints']['point']
                    


                for joint in person_joint_list_raw:
                    if not has({'is_visible'}, joint): 
                        joint['is_visible'] = False

                    if isinstance(joint['is_visible'], str):
                        is_visible = '1' in joint['is_visible']
                    elif isinstance(joint['is_visible'], list):
                        is_visible = False
                    elif isinstance(joint['is_visible'], bool):
                        is_visible = joint['is_visible']

                    person_object['joints'].append({
                        'x': int(joint['x'] / scale_factor),
                        'y': int(joint['y'] / scale_factor),
                        'id': int(joint['id']),
                        'is_visible': is_visible
                    })
                image_object_people.append(person_object)
            
            if not exclude:
                image_object['image_shape'] = get_image_shape_dict(old_image_path, image_object['image_name'], scale_factor)
                image_object['people'] = image_object_people
                image_object['has_people'] = len(image_object['people']) != 0
                processed_json.append(image_object)

    for image_object in processed_json: # this will calculate the bbouxes
        for person_object in image_object['people']:
            person_object['bbox'] = get_bbox(person_object, image_object)

    print('Saving processed json file')

    with open(processedjson_path, 'w+') as fp:
        json.dump(processed_json, fp)

    print('Processing finished!')

    num_original = len(raw_json['annolist'])
    num_processed = len(processed_json)

    print('! Recovered {:2.2f}% of the images (processed {:1} out of {:1})'.format(
        100 * num_processed / num_original,
        num_processed,
        num_original
    ))


    

if __name__ == '__main__': 
    main()