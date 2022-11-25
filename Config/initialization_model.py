import torch
import torch.nn as nn
from typing import cast, Any, Union, Dict, List
from keras.models import load_model
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

class VGG(nn.Module):
    def __init__(self, features:nn.Module, num_classes:int=1000, init_weights:bool=True):
        super(VGG, self).__init__()
        self.features = features #features = Feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self.initialize_weights()
    def forward(self, x):
        x = self.features(x) #features = Feature extraction
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def make_layers(cfg:List[Union[int,str]], batch_norm:bool=False) -> nn.Sequential:
    layers:List[nn.Module] = []
    in_channels = 3
    in_padding = 5
    i = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            in_padding = 1
            if i == 5:
                in_padding = 2
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=in_padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU(num_parameters=1)]
            else:
                layers += [conv2d, nn.PReLU(num_parameters=1)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[int, str]]] = {
    'A': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M'],
    'B': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 1024, 'M'],
    'firstPadding':2
}

def selfDefineVgg(arch, cfg, batch_norm,  num_classes:int, **kwargs: Any) -> VGG:
    model = VGG(make_layers(arch[cfg], batch_norm=batch_norm), num_classes, **kwargs)
    return model

def Init_Main_Model_PT(device: torch.device):
    model_path = '../models/pytorch'
    model_name = 'model=2022-11-25_08-28-04-0.9051.pth'

    model_main = selfDefineVgg(cfgs, 'A', True, 11)
    model_main = model_main.to(device)

    state_dict = torch.load(model_path + '/' + model_name)
    model_main.load_state_dict(state_dict=state_dict)
    return model_main

def Init_Wear_Model_PT(device: torch.device):
    model_path = '../models/pytorch'
    model_name = 'model=2022-11-24_19-41-55-0.9850.pth'

    model_wear = selfDefineVgg(cfgs, 'B', True, 2)
    model_wear = model_wear.to(device)

    state_dict = torch.load(model_path + '/' + model_name)
    model_wear.load_state_dict(state_dict=state_dict)
    return model_wear

def Init_Model_Segmentation_TF():
    saved = load_model("../models/tensorflow/save_ckp_frozen.h5")
    return saved