import torch
import torch.nn as nn
import timm

import os

class MyModels():
    def mobilev3(self):
        model =  timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
        model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)

        return model

    def mobilevitv3(self):
        model = timm.create_model('tf_mobilenetv3_large_100.in1k', pretrained=True)
        model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)

        return model

    def efficientnetv2(self):
        model = timm.create_model('tf_efficientnetv2_xl', pretrained=True)
        model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)

        return model

    def densenet201(self):
        model = timm.create_model('densenet201.tv_in1k', pretrained=True)
        model.classifier = nn.Linear(in_features=1920, out_features = 1, bias=True)
        
        return model
