import torch
import torch.nn as nn 
import arch.models as models
from mpii_dataset import *
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

if __name__ == '__main__':
    image_path = './images/mpii_resized'
    json_path = './annotations/mpii/fullannotations.json'

    dataset = MPIIDataset(json_path, image_path)

    # These values are optimal for me!
    torch_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, collate_fn=lambda x: x)


    for batch in tqdm(torch_dataloader):
        for (image, annotation) in batch:
            pass