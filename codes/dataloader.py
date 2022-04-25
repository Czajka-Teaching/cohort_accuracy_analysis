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






############ Checks ######################

# path_txt = "/afs/crc.nd.edu/user/a/abhatta/MLproject/MLbalancedproject/preprocessingMLproject/txtfiles/check.txt"

# #transform for this dataset
# transform = transforms.Compose([
# transforms.Resize((256,256)),
# transforms.CenterCrop((224,224)),
# transforms.ToTensor(),
# transforms.Normalize(mean=[0.6527, 0.4830, 0.4047], std=[0.2358, 0.2069, 0.1974])])

# loader = ImageList(path_txt,transform=transform)

# images, labels = next(iter(loader))

# one_batch = next(iter(loader))

# print(one_batch.shape)
