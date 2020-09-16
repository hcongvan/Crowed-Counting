import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import json

class CSRNet(nn.Module):
    def __init__(self, cfg_file, use_pretrain=True):
        super(CSRNet,self).__init__()
        if isinstance(cfg_file,str):
            with open(cfg_file) as f:
                cfg = json.load(f)
        self.config_model = cfg.get('CSRNet',None)
        if use_pretrain:
            self.frontend = self.build_model(self.config_model.get('frontend'),3)
            self.backend = self.build_model(self.config_model.get('backend'),512)
            vgg16 = models.vgg16(pretrained=True)
            b = list(self.frontend.state_dict().items())
            a = list(vgg16.state_dict().items())
            value = a[0][1]
            for i in range(len(b)):
                b[i][1].copy_(a[i][1])

            
        else:
            self.frontend = self.build_model(self.config_model.get('frontend'),3)
            self.backend = self.build_model(self.config_model.get('backend'),512)

    def build_model(self,cfg,input_channels):
        layers = []
        in_shape = input_channels
        for layer in cfg:
            if layer.get('type') == 'conv2d':
                layers.append(
                    nn.Conv2d(
                        in_shape,
                        layer.get('out_channels'),
                        layer.get('kernel',3),
                        layer.get('strides',1),
                        layer.get('padding',1),
                        layer.get('dilation',1)
                    )
                )
            if layer.get('type') == 'maxpooling':
                layers.append(
                    nn.MaxPool2d(
                        layer.get('kernel',2),
                        layer.get('strides',2)
                    )
                )
            if layer.get('activation',None) is not None:
                layers.append(nn.ReLU(inplace=True))
            in_shape = layer.get('out_channels',in_shape)

        return nn.Sequential(*layers)
    
    def forward(self,x):
        bs,c,h,w = x.size()
        x = self.frontend(x)
        x = self.backend(x)
        return x

# inputX = torch.randn(3,3,240,240)
# model = CSRNet('model.json')
# out = model(inputX)
# print("TEST")