import torch
import torch.nn as nn 
import json 
import src.mpii_dataset as ds 
import src.arch.models as models 
import src.arch.loss as loss 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

FULL_ANNOTATION_PATH = './annotations/mpii/fullannotations.json'
TABLE_ANNOTATION_PATH = './annotations/mpii/jointtable.json'
TABLE_TENSOR_PATH = './images/mpii_table/jointtable.pt'
IMAGE_DIR = './images/mpii_resized/'
TRAIN_DIR = './train/'
CUDA = torch.device('cuda')

TRAIN_BATCH = 128
VAL_BATCH = 128
TEST_BATCH = 128

PERCENT_TRAIN = 0.7
PERCENT_VAL = 0.1
PERCENT_TEST = 0.2

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
saved_models = os.listdir(TRAIN_DIR)
if len(saved_models) == 0:
    model = models.SinglePersonPoseEtimator()
    criterion = loss.CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    model.to(CUDA)


for batch_id, batch in tqdm(enumerate(train_dataloader)):

    for image_tensor, truth_tables, full_annotation_object, table_annotation_object in batch:
        pass
        # send the data to cuda!

        # train the network here!
    

    # print(value)