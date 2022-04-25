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
from dataloader import ImageList


def test(test_imgpath,saved_model_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5069, 0.4518, 0.4377], std=[0.2684, 0.2402, 0.2336])])

    print("Testing Results on the MORPH")
    print("Test DataLoading Started!")
    start = time.time()
    # Loading the validation data
    val_dataset = ImageList(image_list=test_imgpath,transform=transform)
    valloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                            shuffle=True, num_workers=8,pin_memory=True)
    print("Validation DataLoading Ended!")
    end = time.time()
    print("Total time taken for Train Data Loading:",end-start)
    print("The length of the validation dataset is:",len(val_dataset))

    # check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet.ResNet50(2)
    #loss function and optimizers instantiation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)

    print("Loading the pre-trained weights......")

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(saved_model_path))
    else:
        model.load_state_dict(torch.load(saved_model_path))

    #model.load_state_dict(torch.load("../saved_models/resnet{}_model.pth".format(resnet_layer_num)))
    model.eval()
    model.to(device)

    #Validation Loop
    correct_pred_val = 0
    with torch.no_grad():
        # set the model in evaluation mode, so batchnorm and dropout layers work in eval mode
        model.eval()
        val_loss = 0
        # loop over the validation set
        for data, labels in tqdm(valloader):
            (data, labels) = (data.to(device), labels.to(device))
            outputs = model(data)
            #validation loss
            loss = criterion(outputs,labels)
            val_loss += loss.item()

            _,pred = torch.max(outputs, 1)
            correct_pred_val += torch.sum(pred==labels)
        
        val_acc = (correct_pred_val/len(val_dataset))*100
        print("The total number of correct_prediction_is:",correct_pred_val)
        print("Accuracy for the validation set = {}". format(val_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a test on ResNet-50 trained model")
    parser.add_argument("--testlist", "-t", required = True, help="path to txt file with test list")
    parser.add_argument("--path", "-p", required = True, help="path to the saved models")
  
    args = parser.parse_args()
    
    test(test_imgpath=args.testlist,saved_model_path=args.path)
