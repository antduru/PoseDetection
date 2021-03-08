import torch
import torch.nn as nn 
import json 
import src.mpii_dataset as ds 
import src.arch.models as models 
# import src.arch.loss as loss 
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

TRAIN_BATCH = 128 # default 128
VAL_BATCH = 128 # default 128
TEST_BATCH = 128 # default 128

PERCENT_TRAIN = 0.27 # normally 70%
PERCENT_VAL = 0.03  # normally 10%
PERCENT_TEST = 0.7

LR = 0.01

LOSS_A = 0.5

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
criterion = models.CustomLoss()

model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
train_loss_array = []
val_loss_array = []

if len(previous_saves) != 0:
    saves = [int(e.replace('snapshot-', '').replace('.pt', '')) for e in previous_saves]

    latest_save = max(saves)
    print(f'Founded snapshot! Loading snapshot-{latest_save}.pt')
    status_quo = torch.load(os.path.join(TRAIN_DIR, f'snapshot-{latest_save}.pt'))
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
    # old_loss = 0
    for batch_id, batch in enumerate(tqdm(train_dataloader)):

        total_train_loss = 0
        count = 0

        for image_tensor, truth_tables, full_annotation_object, table_annotation_object in batch:
            truth_tables = truth_tables.cuda()
            image_tensor = image_tensor.cuda()
            for bbox_index, truth_table in enumerate(truth_tables):
                detected_bounding_box = full_annotation_object['people'][bbox_index]['bbox'] # this will be given by yolov3

                optimizer.zero_grad()
                output_table, bbox = model(image_tensor, detected_bounding_box)
                loss = criterion(output_table, truth_table)
                loss.backward()
                optimizer.step()

                # if False:
                #     old_loss = LOSS_A * float(loss) + (1 - LOSS_A) * old_loss
                #     print('loss: {:2.2f}'.format(old_loss))

                count += 1
                total_train_loss += float(loss)

        train_loss_array.append({
            'avg_loss': total_train_loss / count,
            'time': str(datetime.now()),
            'epoch': epoch,
            'batch': batch_id
        })
            
        model.eval()

        total_val_loss = 0
        count2 = 0

        for batch_id, batch in enumerate((val_dataloader)):

            for image_tensor, truth_tables, full_annotation_object, table_annotation_object in batch:
                truth_tables = truth_tables.cuda()
                image_tensor = image_tensor.cuda()

                for bbox_index, truth_table in enumerate(truth_tables):
                    detected_bounding_box = full_annotation_object['people'][bbox_index]['bbox'] # this will be given by yolov3

                    with torch.no_grad():
                        output_table, bbox = model(image_tensor, detected_bounding_box)
                        loss = criterion(output_table, truth_table)

                    count2 += 1
                    total_val_loss += float(loss)

        val_loss_array.append({
            'avg_loss': total_val_loss / count2,
            'time': str(datetime.now()),
            'epoch': epoch,
            'batch': batch_id
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