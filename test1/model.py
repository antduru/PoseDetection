import torch
import torch.nn as nn 
import torch.nn.functional as F 
import collections
import copy
import torchvision

class CustomVgg19(nn.Module):
    def __init__(self, original: torchvision.models.VGG, k=10):
        ''' get first k including k's layer -> 0 to 36 '''
        super().__init__()
        features = []

        for i, module in enumerate(next(original.children())):
            features.append(module)

        self.features = nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)




with torch.no_grad():
    model = torchvision.models.vgg19(pretrained=True)
    custom_model = CustomVgg19(model)

    model.eval()
    custom_model.eval()

    image = torch.zeros((1, 3, 256, 256))

    print(model(image).shape)
    print(custom_model(image).shape)
