from shutil import copyfile
import json 
import os 
from tqdm import tqdm

old_image_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/images'
new_image_path = './images/mpii'
json_path = './annotations/mpii/fullannotations.json'

with open(json_path, 'r') as fp:
    json_file = json.load(fp)

for json_object in tqdm(json_file):
    image_name = json_object['image_name']
    old_image_file = os.path.join(old_image_path, image_name)
    new_image_file = os.path.join(new_image_path, image_name)

    copyfile(old_image_file, new_image_file)