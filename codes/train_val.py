import resnet as ResNet
import config
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import time
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from dataloader import ImageList
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/")

def train_val(resnet_layer_num,train_imglist,val_imglist,save_path):

    #transform for this validation dataset - Kaggle DataSet
    transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6527, 0.4830, 0.4047], std=[0.2358, 0.2069, 0.1974]),])

    #transform for this Training dataset - MS1MV2 DataSet
    transform_train = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
            
    
    ########### DataLoading for Training ############# --> Both training and validation data from kaggle dataset for this instance
    print("Train DataLoading Started!")
    start = time.time()
    # Loading the training data
    train_dataset = ImageList(image_list=train_imglist,transform=transform_train)
    trainloader = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True, num_workers=8,pin_memory=True)
    print("Train DataLoading Ended!")
    end = time.time()
    print("Total time taken for Train Data Loading:",end-start)
    print("The length of the training dataset is:",len(train_dataset))


    ########### DataLoading for Validation ##############
    print("Validation DataLoading Started!")
    start = time.time()
    # Loading the validation data
    val_dataset = ImageList(image_list=val_imglist,transform=transform_val)
    valloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                            shuffle=True, num_workers=8,pin_memory=True)
    print("Validation DataLoading Ended!")
    end = time.time()
    print("Total time taken for Train Data Loading:",end-start)
    print("The length of the training dataset is:",len(val_dataset))

    
    # check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiation of the model class based on layer_num
    if resnet_layer_num == 50:
        model = ResNet.ResNet50(2)
    elif resnet_layer_num == 101:
        model = ResNet.ResNet101(2)
    elif resnet_layer_num == 152:
        model = ResNet.ResNet152(2)
    
    if torch.cuda.device_count() > 1:
        print(f"Model will use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    #loss function and optimizers instantiation
    criterion =  nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=10)
    #scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1,verbose=True)
    best_acc = 0
    best_accuracy_epoch = 0
    #Training loop
    for epochs in range(config.epochs):
        start_train = time.time()
        model.train()
        training_loss = 0
        correct_pred_train = 0
        print("EPOCH Training {} Started!".format(epochs+1))
        for item,(data,labels) in enumerate(trainloader):

            #print(labels)
            (data,labels) = (data.to(device),labels.to(device))

            optimizer.zero_grad()
            outputs = model(data)

            #print("Outputs",outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            _,pred = torch.max(outputs, dim = 1)

            correct_pred_train += torch.sum(pred == labels) # (pred == labels).sum().item() #torch.sum(pred==labels.squeeze(1))

        
        end_train = time.time()

        train_acc = (correct_pred_train/len(train_dataset))*100
        print("Time taken to finish this Epoch is",(end_train-start_train)/3600)
        print("Loss at the end of Epoch {} = {}".format(epochs+1,training_loss))
        print("Training Accuracy at the end of Epoch {} = {}". format(epochs+1,train_acc))

        #Validation Loop
        correct_pred_val = 0
        with torch.no_grad():
            # set the model in evaluation mode, so batchnorm and dropout layers work in eval mode
            model.eval()
            val_loss = 0
            # loop over the validation set
            for data, labels in valloader:
                (data, labels) = (data.to(device), labels.to(device))
                outputs = model(data)
                #validation loss
                loss = criterion(outputs,labels)
                val_loss += loss.item()

                _,pred = torch.max(outputs, 1)
                correct_pred_val += torch.sum(pred==labels)
            
            val_acc = (correct_pred_val/len(val_dataset))*100
            print("Accuracy for the validation set = {}". format(val_acc))

            if val_acc > best_acc:
                best_accuracy_epoch = epochs
                print("Saving the best model so far at {}".format(save_path))
                #Saving this way allows model to be later loaded on multi/single GPU
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(save_path,"resnet{}_model.pth".format(resnet_layer_num)))
                else:
                    torch.save(model.module.state_dict(), os.path.join(save_path,"resnet{}_model.pth".format(resnet_layer_num)))
                #torch.save(model.state_dict(),"/afs/crc.nd.edu/user/a/abhatta/NNproject/saved_models/resnet{}_model.pth".format(resnet_layer_num))
                best_acc = val_acc
            
            scheduler.step(val_loss)
        
        print("The best validation accuracy so far is {} and occured at {}".format(best_acc,best_accuracy_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate ResNet-models ")
    parser.add_argument("--layernum", "-n", required = True, type =int, help="Number of ResNet Layer")
    parser.add_argument("--trainlist", "-t", required = True, help="path to txt file with train list")
    parser.add_argument("--vallist", "-v", required = True, help="path to txt file with validation list")
    parser.add_argument("--dest", "-d", required = True, help="path to save the trained_model")
    args = parser.parse_args()
    

    # Asserting the Resnet model can only be of 50,100 or 152 layers
    assert args.layernum in [50,101,152]


    train_val(resnet_layer_num = args.layernum,train_imglist=args.trainlist,val_imglist=args.vallist,save_path=args.dest)

    train_val(args.layernum)
