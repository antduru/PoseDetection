import torch
import torch.nn as nn 
import torch.nn.functional as F 
import collections
import copy
import torchvision
import models


with torch.no_grad():
    cropped_model = models.CroppedVgg19().eval()

    image = torch.zeros((1, 3, 256, 256))

    print(cropped_model(image).shape)
