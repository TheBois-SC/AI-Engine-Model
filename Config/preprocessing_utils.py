from torchvision.transforms import functional
from PIL import ImageStat
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from rembg import remove
from io import BytesIO, StringIO
import base64
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from Config.initialization_model import VGG


# Show Function for image tensor
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def is_grayscale(path):
    im = Image.open(BytesIO(base64.b64decode(path))).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum) / 3 == stat.sum[0]:  # check the avg with any element value
        return True  # if grayscale
    else:
        return False  # else its colour


def resize_with_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)
    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


def CropObject(IMG: Image):
    im = IMG
    p, l = im.size
    base_ori = Image.new('RGB', (p, l), (255, 255, 255))
    output = remove(im)
    base_ori.paste(output, (0, 0), output)
    base_ori = base_ori.convert("RGB")
    p, l = base_ori.size
    pix = base_ori.load()
    termination = [(0, 0, 0), (255, 255, 255)]

    hasil = -1
    for y in range(0, l):
        for x in range(0, p):
            if (pix[x, y] not in termination):
                hasil = y
                break
        if (hasil != -1):
            break
    left = 0;
    upper = hasil;
    right = p;
    lower = l
    ori_crop1 = base_ori.crop((left, upper, right, lower))

    hasil = -1
    pix = ori_crop1.load()
    p, l = ori_crop1.size
    for y in range(l - 1, 0, -1):
        for x in range(0, p):
            if (pix[x, y] not in termination):
                hasil = y
                break
        if (hasil != -1):
            break
    left = 0;
    upper = 0;
    right = p;
    lower = l - (l - hasil)
    ori_crop2 = ori_crop1.crop((left, upper, right, lower))

    hasil = -1
    pix = ori_crop2.load()
    p, l = ori_crop2.size
    for x in range(0, p):
        for y in range(0, l):
            if (pix[x, y] not in termination):
                hasil = x
                break
        if (hasil != -1):
            break
    left = hasil;
    upper = 0;
    right = p;
    lower = l
    ori_crop3 = ori_crop2.crop((left, upper, right, lower))

    hasil = -1
    pix = ori_crop3.load()
    p, l = ori_crop3.size
    for x in range(p - 1, 0, -1):
        for y in range(0, l):
            if (pix[x, y] not in termination):
                hasil = x
                break
        if (hasil != -1):
            break
    left = 0;
    upper = 0;
    right = p - (p - hasil);
    lower = l
    ori_crop4 = ori_crop3.crop((left, upper, right, lower))

    return ori_crop4


class fashion_tools_segmentation(object):
    def __init__(self, imageid, model, version=1.1):
        self.imageid = imageid
        self.model = model
        self.version = version

    def readb64(base64_string):
        sbuf = StringIO()
        sbuf.write(base64.b64decode(base64_string))
        pimg = Image.open(sbuf)
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)

    def get_fashion(self, stack=False):
        """limited to top wear and full body dresses (wild and studio working)"""
        """takes input rgb----> return PNG"""
        name = self.imageid
        file = self.readb64(self.imageid)
        file = tf.image.resize_with_pad(file, target_height=512, target_width=512)
        rgb = file.numpy()
        file = np.expand_dims(file, axis=0) / 255.
        seq = self.model.predict(file)
        seq = seq[3][0, :, :, 0]
        seq = np.expand_dims(seq, axis=-1)
        c1x = rgb * seq
        c2x = rgb * (1 - seq)
        cfx = c1x + c2x
        dummy = np.ones((rgb.shape[0], rgb.shape[1], 1))
        rgbx = np.concatenate((rgb, dummy * 255), axis=-1)
        rgbs = np.concatenate((cfx, seq * 255.), axis=-1)
        if stack:
            stacked = np.hstack((rgbx, rgbs))
            return stacked
        else:
            return rgbs


def DetectWearing(path: str, model: VGG, device: torch.device):
    img = Image.open(BytesIO(base64.b64decode(path))).convert("RGB")
    transform_norm_rgb = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.ToTensor()
    ])
    img_normalized = transform_norm_rgb(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)

    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = [False, True]  # ["No Wearing", "Wearing"]
        class_name = classes[index]
        return class_name
