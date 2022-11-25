import torch
import torch.nn as nn
from torchvision.transforms import transforms
from typing import cast, Any, Union, Dict, List
from PIL import Image, ImageOps
import numpy as np
from rembg import remove
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import tensorflow as tf

from Config.initialization_model import VGG
from Config.preprocessing_utils import (
    DetectWearing,
    is_grayscale,
    fashion_tools_segmentation,
    CropObject,
    resize_with_pad
)

def DetectingFashion(path: str, model_main: VGG, model_wear: VGG, device: torch.device, tf_model: load_model):
    if (is_grayscale(path)):
        ori = Image.open(path).convert("RGB")
        main_color = max(ori.getcolors(ori.size[0]*ori.size[1]))[1]
        if (main_color != (0,0,0)):
            transGrayscale1 = transforms.Compose([
                transforms.Resize((70,70)),
                transforms.RandomInvert(p=1),
                transforms.ToTensor()
            ])
            img_normalized = transGrayscale1(ori).float()
            img_normalized = img_normalized.unsqueeze_(0)
            img_normalized = img_normalized.to(device)
        else:
            transGrayscale2 = transforms.Compose([
                transforms.Resize((70,70)),
                transforms.ToTensor()
            ])
            img_normalized = transGrayscale2(ori).float()
            img_normalized = img_normalized.unsqueeze_(0)
            img_normalized = img_normalized.to(device)
    else:
        if DetectWearing(path=path, model=model_wear):
            api = fashion_tools_segmentation(path, tf_model)
            image_ = api.get_dress(stack=False)
            cv2.imwrite("../images/Segmentation_Result",image_)
            ori = Image.open("../images/Segmentation_Result/out.png")
            base_ori = Image.new('RGB', (ori.size[0], ori.size[1]), (255, 255, 255))
            base_ori.paste(ori, (0,0), ori)
            cropped = CropObject(IMG=base_ori)
            imgS = resize_with_pad(im=cropped, target_width=500, target_height=500)
        else:
            ori = Image.open(path).convert("RGB")
            p, l = ori.size
            base_ori = Image.new('RGB', (p, l), (255, 255, 255))
            output = remove(ori)
            base_ori.paste(output, (0,0), output)
            cropped = CropObject(IMG=base_ori)
            imgS = resize_with_pad(im=cropped, target_width=500, target_height=500)

        img = imgS.convert('L').convert('RGB')

        transform_norm_rgb = transforms.Compose([
            transforms.RandomInvert(p=1),
            transforms.Resize((70,70)),
            transforms.ToTensor()
        ])
        img_normalized = transform_norm_rgb(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        img_normalized = img_normalized.to(device)

    with torch.no_grad(): # Predict Image
        model_main.eval()
        output =model_main(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = ["Ankle Boot", "Bag", "Coat", "Dress", "Hat", "Pullover", "Sandal", "Shirt", "Sneaker", "T-shirt/Top", "Trouser"]
        class_name = classes[index]
        return class_name