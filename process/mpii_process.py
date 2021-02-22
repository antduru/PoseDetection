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

def has(sett, dictt):
    n = 0
    for sette in sett:
        if not sette in dictt: continue
        n += 1
    return n == len(sett)

def main():
    rawjson_path = './annotations/mpii/raw_annotations.json'
    processedjson_path = './annotations/mpii/fullannotations.json'

    print('Reading raw json file')

    with open(rawjson_path) as fp:
        raw_json = json.load(fp)['RELEASE']

    print('Processing raw json file')

    processed_json = []
    person_should_include = {'annopoints', 'x1', 'y1', 'x2', 'y2', 'scale'}

    for anno, is_train in zip(raw_json['annolist'], raw_json['img_train']):
        if is_train == 0:
            continue
        else:
            image_object = {
                'image_name': anno['image']['name'],
                'is_train': True,
                "has_people": False,
                'people': []
            }

            for person in anno['annorect']:
                if not has(person_should_include, person): continue
                    
                person_object = {
                    'head_coordinates': {
                        "x1": int(person['x1']), 
                        "y1": int(person['y1']), 
                        "x2": int(person['x2']), 
                        "y2": int(person['y2'] )
                    },
                    'joints': []
                }
                
                if len(person['annopoints']) == 0: continue # no one in the image
                
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
                        'x': int(joint['x']),
                        'y': int(joint['y']),
                        'id': int(joint['id']),
                        'is_visible': is_visible
                    })
                image_object['people'].append(person_object)

            image_object['has_people'] = len(image_object['people']) == 0
            processed_json.append(image_object)

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