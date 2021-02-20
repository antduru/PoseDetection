import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from collections import OrderedDict
import copy

def generate(block_list):
    length = len(block_list)
    ll = []

    for module_name, params in block_list:
        if 'conv' in module_name: ll.append(nn.Conv2d(*params))
        elif 'pool' in module_name: ll.append(nn.MaxPool2d(*params))
        elif 'relu' in module_name: ll.append(nn.ReLU())
    
    return nn.Sequential(*ll)

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
        # pool (kernel, padding, stride)
        # relu ()

        stages = {}

        self.stage0 = generate([
            ('conv', (512, 256, 3, 1, 1)), # block 1
            ('relu', (0,)),
            ('conv', (256, 256, 3, 1, 1)), # block 2
            ('relu', (0,)),
            ('conv', (256, 256, 3, 1, 1)), # block 3
            ('relu', (0,)),
            # ('pool', (2, 1, 0)),
            ('conv', (256, 6, 3, 1, 1)), # block 4
        ])

        self.stage1 = generate([
            ('conv', (512+6, 128, 3, 1, 1)), # block 1
            ('relu', (0,)),
            ('conv', (128, 128, 3, 1, 1)), # block 2
            ('relu', (0,)),
            ('conv', (128, 128, 3, 1, 1)), # block 3
            ('relu', (0,)),
            ('conv', (128, 512, 1, 1, 0)), # block 4
            ('relu', (0,)),
            ('conv', (512, 6, 1, 1, 0)),   # block 5
        ])

        self.stage2 = generate([
            ('conv', (512+6, 128, 7, 1, 3)),   # block 1
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 2
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 3
            ('relu', (0,)),
            ('conv', (128, 512, 1, 1, 0)), # block 4
            ('relu', (0,)),
            ('conv', (512, 6, 1, 1, 0)),   # block 5
        ])

        self.stage3 = generate([
            ('conv', (512+6, 128, 7, 1, 3)),   # block 1
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 2
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 3
            ('relu', (0,)),
            ('conv', (128, 512, 1, 1, 0)), # block 4
            ('relu', (0,)),
            ('conv', (512, 6, 1, 1, 0)),   # block 5
        ])

        self.stage4 = generate([
            ('conv', (512+6, 128, 7, 1, 3)),   # block 1
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 2
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 3
            ('relu', (0,)),
            ('conv', (128, 512, 1, 1, 0)), # block 4
            ('relu', (0,)),
            ('conv', (512, 6, 1, 1, 0)),   # block 5
        ])

        self.stage5 = generate([
            ('conv', (512+6, 128, 7, 1, 3)),   # block 1
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 2
            ('relu', (0,)),
            ('conv', (128, 128, 7, 1, 3)), # block 3
            ('relu', (0,)),
            ('conv', (128, 512, 1, 1, 0)), # block 4
            ('relu', (0,)),
            ('conv', (512, 6, 1, 1, 0)),   # block 5
        ])
    
    def forward(self, in1):
        
        out0a = self.stage0(in1)
        out0 = torch.cat([out0a, in1], 1)

        out1a = self.stage1(out0)
        out1 = torch.cat([out1a, in1], 1)

        out2a = self.stage2(out1)
        out2 = torch.cat([out2a, in1], 1)

        out3a = self.stage3(out2)
        out3 = torch.cat([out3a, in1], 1)

        out4a = self.stage4(out3)
        out4 = torch.cat([out4a, in1], 1)

        out5a = self.stage5(out4)
        out5 = torch.cat([out5a, in1], 1)

        return out5

if __name__ == '__main__':
    cnn1 = CNN1()

    image = torch.zeros((1, 512, 120, 120))

    out = cnn1(image)

    print(out.shape)

