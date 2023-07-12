import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from degradation.degradation import Degradation
from utils import check_image_file
import cv2
class duqu_tupian(object):
    def __init__(self, image_size):
        self.image_size = image_size
    def __call__(self, image_name):
        crop_pad_size = self.image_size
        img_gt = cv2.imread(image_name)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        w,h = img_gt.shape[0:2]
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_w, 0, pad_h, cv2.BORDER_REFLECT_101)
        img_gt=Image.fromarray(img_gt)
        return img_gt

class Dataset(object):
    def __init__(self, images_dir, image_size, upscale_factor):
        deg = Degradation(upscale_factor)
        self.image_size = image_size
        self.filenames = [
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if check_image_file(x)
        ]
        self.lr_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(deg.degradation_pipeline),
                transforms.ToTensor(),
            ]
        )
        self.hr_transforms = transforms.Compose(
            [   
                duqu_tupian(image_size),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        img_hr = self.filenames[idx]
        # pad
        hr = self.hr_transforms(img_hr)
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
