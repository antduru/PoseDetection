import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from collections import OrderedDict
import copy

def generate(block_list):
    length = len(block_list)

    for i in range(1, length + 1):
        if i == length: 


class CroppedVgg19(nn.Module):
    def __init__(self, k=10):
        ''' 
            Get first k including k's layer -> 0 to 36
            k=10 gives 512 channels
        '''
        original = torchvision.models.vgg19(pretrained=True)
        super().__init__()
        features = []

        for i, module in enumerate(next(original.children())):
            features.append(module)

        self.features = nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        # conv (in, out, kernel, padding, stride)

        self.adapter = generate([
            ('conv', (128, 128, 3, 1, 1)), # conv 1
            ('conv', (128, 128, 3, 1, 1)),  # conv 2
            ('conv', (128, 128, 3, 1, 1)),  # conv 3
            ('conv', (128, 128, 3, 1, 1)),  # conv 4
            ('conv', (128, 128, 3, 1, 1)),  # conv 5
        ])

    
    def forward(self, x):
        return self.vgg19_10(x)

if __name__ == '__main__':
    cnn1 = CNN1()

    image = torch.zeros((1, 3, 120, 120))

    out = cnn1(image)

    print(out.shape)

