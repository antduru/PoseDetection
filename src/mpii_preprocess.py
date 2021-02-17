import torch 
import torchvision
from scipy.io import loadmat
from tqdm import tqdm 
import json

def main():    
    def get_all_as_json(tuple_1, tuple_2):
        peopel_as_json = {}

        i = 0

        for _row, _row2 in tqdm(zip(tuple_1[0], tuple_2[0])):
            image_name, annorect, vidx, frame_set = _row
            image_name = image_name[0, 0].item()[0][0]

            peopel_as_json[image_name] = {
                'train': bool(_row2 == 1),
                'people': get_image_anotation(annorect)
            }

            # i += 1
            if i == 5: break 
        
        return peopel_as_json        

    def get_image_anotation(annorect):
        annotated_peope = []

        for person_row in annorect[0]:
            if len(person_row.item()) != 7: continue

            x1, y1, x2, y2 = person_row.item()[:4]
            x1 = x1[0, 0]
            y1 = y1[0, 0]
            x2 = x2[0, 0]
            y2 = y2[0, 0]
            scale_wrt200 = person_row.item()[5][0, 0]
            object_pos = person_row.item()[6]
            posx, posy = object_pos.item()
            posx = posx[0, 0]
            posy = posy[0, 0]
            annotpoints = person_row.item()[4]
            joints = []

            for row in annotpoints.item()[0][0]:
                x, y, joint_id, visible = row
                x = x[0, 0]
                y = y[0, 0]
                joint_id = joint_id[0, 0]
                if visible.shape == (1, 1):
                    visible = visible[0, 0] == 1
                else:
                    visible = False

                joints.append({
                    'x': int(x),
                    'y': int(y),
                    'id': int(joint_id),
                    'is_visible': bool(visible)
                })

            annotated_person = {
                'head_coordinates': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                },
                'scale': float(scale_wrt200),
                'object_position': {
                    'x': int(posx),
                    'y': int(posy)
                },
                'anno_points': joints
            }
            annotated_peope.append(annotated_person)
        return annotated_peope

    images_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/images'
    matlab_path = '/media/ubombar/Backup/datasets/original/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    # json_result_path = '/media/ubombar/Backup/datasets/processed/mpii_preprocessed/result_all.json'
    json_result_path = './result_all.json'

    matlab_file = loadmat(matlab_path)

    t1, t2, t3, t4, t5, t6 = matlab_file['RELEASE'][0, 0].item()
    everything = get_all_as_json(t1, t2)

    with open(json_result_path, 'w+') as fp:
        json.dump(everything, fp, indent=4)

    print('done!')


main()