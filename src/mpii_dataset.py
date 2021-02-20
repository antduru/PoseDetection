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

# @DeprecationWarning('I wrote this to test the speed no not use this!')
# class CustomDataloader():
#     '''
#         I wrote this to test the speed no not use this!
#     '''
#     def __init__(self, dataset:Dataset, bounds:tuple, batch_size:int=1, shuffle:bool=False):
#         assert batch_size > 0
#         assert len(bounds) == 2

#         self.dataset = dataset
#         self.first, self.last = bounds
#         self.order = [i for i in range(self.first, self.last, 1)]

#         if shuffle:
#             random.shuffle(self.order)
        
#         self.batch_i = 0
#         self.batch_size = batch_size
    
#     def __iter__(self):
#         self.batch_i = 0
#         return self

#     def __len__(self):
#         return math.ceil(len(self.order) / self.batch_size)
    
#     def __next__(self):
#         if self.batch_i < len(self):
#             self.batch_i += 1

#             order_slice = self.order[self.batch_size * (self.batch_i - 1): self.batch_size * self.batch_i]

#             return [self.dataset[idx] for idx in order_slice]
#         else:
#             raise StopIteration()

class MPIIDataset(Dataset):
    def __init__(self, json_path, image_path, transform=None):
        '''
            json_path (string): Path to the fullannotations.json
            image_path (string): Path to the image folder
        '''
        with open(json_path, 'r') as fp:
            self.json_file = json.load(fp)
        
        self.image_path = image_path
        self.transform = transform
        self.default_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.json_file)
    
    # def __getitem__(self, idx):

    #     if isinstance(idx, int):
    #         idx = [idx]

    #     annotations = itemgetter(*idx)(self.json_file)

    #     if not isinstance(annotations, list):
    #         annotations = [annotations]
        
    #     image_path = [path.join(self.image_path, anno['image_name']) for anno in annotations] 

    #     if self.transform:
    #         images = [self.transform(self.default_transform(Image.open(fn))) for fn in image_path]
    #     else:
    #         images = [self.default_transform(Image.open(fn)) for fn in image_path]

        
    #     return {'images': images, 'annotations': annotations}

    def __getitem__(self, idx:int):
        annotation = self.json_file[idx]
        image_path = path.join(self.image_path, annotation['image_name'])
        image = self.default_transform(Image.open(image_path))

        if self.transform:
            image = self.transform(image)

        return (image, annotation)


if __name__ == '__main__':
    image_path = './images/mpii_resized'
    json_path = './annotations/mpii/fullannotations.json'

    dataset = MPIIDataset(json_path, image_path)

    # These values are optimal for me!
    torch_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=lambda x: x)


    for batch in tqdm(torch_dataloader):
        for (image, annotation) in batch:
            pass
        