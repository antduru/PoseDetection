from os import path
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json 
from tqdm import tqdm
from operator import itemgetter
from collections.abc import Iterable
import random
import math

def get_bounds(FULL_ANNOTATION_PATH, PERCENT_TRAIN, PERCENT_VAL, PERCENT_TEST):
    num_images = how_many_images(FULL_ANNOTATION_PATH)

    train_bound = (0, int(num_images * PERCENT_TRAIN))
    val_bound = (int(num_images * PERCENT_TRAIN), int(num_images * (PERCENT_VAL + PERCENT_TRAIN)))
    test_bound = (int(num_images * (PERCENT_VAL + PERCENT_TRAIN)), num_images)

    return train_bound, val_bound, test_bound

def how_many_images(full_annotation_path):
    with open(full_annotation_path, 'r') as fp:
        return len(json.load(fp))

class MPIIDataset(Dataset):
    def __init__(self, full_annotation, table_annotation, table_tensor, image_dir, bounds):
        assert len(full_annotation) == len(table_annotation)

        self.full_annotation = full_annotation
        self.table_annotation = table_annotation
        self.table_tensor = table_tensor
        self.minimum, self.maximum = bounds
        self.image_dir = image_dir

        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return (self.maximum - self.minimum)
    
    def __getitem__(self, i):
        assert i >= 0 or i < len(self)
        i = i + self.minimum # mapp the index

        full_annotation_object = self.full_annotation[i]
        table_annotation_object = self.table_annotation[i]

        assert full_annotation_object['image_name'] == table_annotation_object['image_name']
        image_name = full_annotation_object['image_name']

        pil_image = Image.open(path.join(self.image_dir, image_name))
        image_tensor = self.transform(pil_image)

        truth_table_indexes = []

        for full_people, table_people in zip(full_annotation_object['people'], table_annotation_object['people']):
            truth_table_indexes.append(table_people['i'])

        truth_tables = self.table_tensor[truth_table_indexes]

        return image_tensor, truth_tables, full_annotation_object, table_annotation_object



