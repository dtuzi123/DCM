import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class Data_Augmentation:
    def RandomTransfer(x1):

        inputSize = np.shape(x1)[2]

        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
        ])

        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(inputSize, pad_if_needed=True),
        ])

        transform3 = transforms.Compose([
            transforms.Compose(),
            transforms.ToPILImage(),
            transforms.RandomRotation(60),
        ])

        r =  transform2(x1)
        return r
