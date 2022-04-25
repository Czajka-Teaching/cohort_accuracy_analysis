from os import path
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch import dtype
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageList(ImageFolder):

    # the img_list is like : path_to_the_image label --> ~/img_folder/check1.jpg 0
    def __init__(self,image_list,transform=None):
    
        self.image_names = np.loadtxt(image_list,dtype=str)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = Image.open(self.image_names[index][0]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, int(self.image_names[index][1])



