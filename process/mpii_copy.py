from shutil import copyfile
import json 
import os 
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

old_image_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/images'
# new_image_path = './images/mpii'
new_image_path = './images/mpii_resized'
json_path = './annotations/mpii/fullannotations.json'

with open(json_path, 'r') as fp:
    json_file = json.load(fp)

for json_object in tqdm(json_file):
    try:
        image_name = json_object['image_name']
        old_image_file = os.path.join(old_image_path, image_name)
        new_image_file = os.path.join(new_image_path, image_name)

        # copyfile(old_image_file, new_image_file)

        image_tensor = to_tensor(Image.open(old_image_file))
        image_resized = to_pil_image(image_tensor[:, ::2, ::2]) # resize do not change

        # with open(new_image_file, 'w+') as fp: pass

        image_resized.save(new_image_file)
    except:
        pass


