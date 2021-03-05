import torch
import torch.nn as nn 
import json 
import src.mpii_dataset as ds 
import src.arch.models as models 
import src.arch.loss as loss 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import math 

# triangle(0, 
#     0, 
#     math.cos(math.radians(omega1)), 
#     math.sin(math.radians(omega1)), 
#     math.cos(math.radians(omega2)), 
#     math.sin(math.radians(omega2)))

FULL_ANNOTATION_PATH = './annotations/mpii/fullannotations.json'
TABLE_ANNOTATION_PATH = './annotations/mpii/jointtable.json'
TABLE_TENSOR_PATH = './images/mpii_table/jointtable.pt'
IMAGE_DIR = './images/mpii_resized/'
TRAIN_DIR = './train/'
EPOCH = 200
CUDA = torch.device('cuda')

TRAIN_BATCH = 128
VAL_BATCH = 128
TEST_BATCH = 128

PERCENT_TRAIN = 0.7
PERCENT_VAL = 0.1
PERCENT_TEST = 0.2

LR = 0.001

# tens = torch.zeros((1, 1), device=CUDA)


with open(FULL_ANNOTATION_PATH, 'r') as fp:
    full_annotation = json.load(fp)

with open(TABLE_ANNOTATION_PATH, 'r') as fp:
    table_annotation = json.load(fp)

table_tensor = torch.load(TABLE_TENSOR_PATH)

train_bound, val_bound, test_bound = ds.get_bounds(FULL_ANNOTATION_PATH, PERCENT_TRAIN, PERCENT_VAL, PERCENT_TEST)

train_dataset = ds.MPIIDataset(full_annotation, table_annotation, table_tensor, IMAGE_DIR, train_bound)
val_dataset   = ds.MPIIDataset(full_annotation, table_annotation, table_tensor, IMAGE_DIR, val_bound)
test_dataset  = ds.MPIIDataset(full_annotation, table_annotation, table_tensor, IMAGE_DIR, test_bound)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=2, collate_fn=lambda x: x)
val_dataloader   = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=True, num_workers=2, collate_fn=lambda x: x)
test_dataloader  = DataLoader(test_dataset, batch_size=TEST_BATCH, shuffle=True, num_workers=2, collate_fn=lambda x: x)

# load model or create
previous_saves = os.listdir(TRAIN_DIR)
model = models.SinglePersonPoseEtimator()
criterion = loss.CustomLoss()

model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_loss_array = []
val_loss_array = []

if len(previous_saves) != 0:
    latest_save = max(previous_saves)
    print(f'Founded snapshot! Loading {latest_save}')
    status_quo = torch.load(os.path.join(TRAIN_DIR, latest_save))
    # optimizer.load_state_dict(status_quo['optimizer'])
    model.load_state_dict(status_quo['model'])
    train_loss_array = status_quo['train_loss']
    val_loss_array = status_quo['val_loss']

    print('Loaded from snapshot!')
    
model = model.cuda()

for epoch in range(EPOCH):
    print(f'Starting epoch {epoch}')
    print(f'\tTraining...')
    model.train()
    for batch_id, batch in enumerate(tqdm(train_dataloader)):

        for image_tensor, truth_tables, full_annotation_object, table_annotation_object in batch:
            truth_tables = truth_tables.cuda()
            image_tensor = image_tensor.cuda()
            loss_average = 0
            for bbox_index, truth_table in enumerate(truth_tables):
                detected_bounding_box = full_annotation_object['people'][bbox_index]['bbox'] # this will be given by yolov3

                optimizer.zero_grad()
                output_table, bbox = model(image_tensor, detected_bounding_box)
                loss = criterion(output_table, truth_table)
                loss.backward()
                optimizer.step()

                loss_average += float(loss) / len(truth_tables)

            train_loss_array.append({
                'loss': loss.item(),
                'time': str(datetime.now()),
                'epoch': epoch
            })
            
    model.eval()
    print(f'\tValidating...')
    for batch_id, batch in enumerate(tqdm(val_dataloader)):

        for image_tensor, truth_tables, full_annotation_object, table_annotation_object in batch:
            truth_tables = truth_tables.cuda()
            image_tensor = image_tensor.cuda()
            loss_average = 0

            for bbox_index, truth_table in enumerate(truth_tables):
                detected_bounding_box = full_annotation_object['people'][bbox_index]['bbox'] # this will be given by yolov3

                with torch.no_grad():
                    output_table, bbox = model(image_tensor, detected_bounding_box)
                    loss = criterion(output_table, truth_table)


                loss_average += float(loss) / len(truth_tables)

            train_loss_array.append({
                'loss': loss.item(),
                'time': str(datetime.now()),
                'epoch': epoch
            })
    print(f'\tSaving...')
    snapshot = len(os.listdir(TRAIN_DIR))
    new_save = os.path.join(TRAIN_DIR, f'snapshot-{snapshot}.pt')
    model.cpu()
    status_quo = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss_array,
        'val_loss': val_loss_array
    }
    torch.save(status_quo, new_save)
    model.cuda()
    
print('Done!')
    # print(value)