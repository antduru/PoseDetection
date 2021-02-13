import torch
import torch.nn as nn 
import torch.nn.functional as F 
import collections
import copy
import torchvision

class CroppedVgg19(nn.Module):
    def __init__(self, k=10):
        ''' get first k including k's layer -> 0 to 36 '''
        original = torchvision.models.vgg19(pretrained=True)
        super().__init__()
        features = []

        for i, module in enumerate(next(original.children())):
            features.append(module)

        self.features = nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)