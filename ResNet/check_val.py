import resnet as ResNet
import config
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import time
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def val_check(resnet_layer_num,imgpath):

    resnet_layer_num = int(resnet_layer_num)

    # Asserting the Resnet model can only be of 50,100 or 152 layers
    assert resnet_layer_num in [50,101,152], "Can only be one of 50,101,152 layers"

    # Instantiation of the model class based on layer_num
    if resnet_layer_num == 50:
        model = ResNet.ResNet50(2)
    elif resnet_layer_num == 101:
        model = ResNet.ResNet101(2)
    elif resnet_layer_num == 152:
        model = ResNet.ResNet152(2)

    # check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("../saved_models/resnet{}_model.pth".format(resnet_layer_num)))
    model.eval()
    model.to(device)

    #transform for this dataset
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6527, 0.4830, 0.4047], std=[0.2358, 0.2069, 0.1974])])

    img = Image.open(imgpath)
    img = img.convert("RGB")  
    img = transform(img)
    img = torch.reshape(img,(1,3,224,224))
  
    output = model(img)

    _,pred = torch.max(output, dim = 1)

    if pred.item() == 1:
        print("GENDER = MALE!!!")
    else:
        print("GENDER = FEMALE!!!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate ResNet-models ")
    parser.add_argument("--layernum", "-n", required = True, help="Number of ResNet Layer")
    parser.add_argument("--imgpath", "-img", required = True, help="Path to Image")
    args = parser.parse_args()
    
    val_check(args.layernum,args.imgpath)
