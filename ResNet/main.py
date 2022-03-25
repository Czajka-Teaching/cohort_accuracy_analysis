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


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/")

def train_val(resnet_layer_num):

    # Crc specific - for multiple training at one - passing all layers at once and they are passed as path , eg: "50/"
    #resnet_layer_num = resnet_layer_num.split("/")[0]

    resnet_layer_num = int(resnet_layer_num)

    # Asserting the Resnet model can only be of 50,100 or 152 layers
    assert resnet_layer_num in [50,101,152], "Can only be one of 50,101,152 layers"
    
    #transform for Imagenet
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6527, 0.4830, 0.4047], std=[0.2358, 0.2069, 0.1974])])

  
    print("Train DataLoading Started!")
    start = time.time()

    # Loading the training data
    train_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Training/',transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True, num_workers=8,pin_memory=True)

    print("Train DataLoading Ended!")
    end = time.time()
    print("Total time taken for Train Data Loading:",end-start)
    print("The length of the training dataset is:",len(train_dataset))


    print("Validation DataLoading Started!")
    start = time.time()
    # Loading the validation data
    val_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Validation/',transform=transform)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,weight_decay=config.weight_decay)
    scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1,verbose=True)
    best_acc = 0
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

        scheduler.step()
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
            # loop over the validation set
            for data, labels in valloader:
                (data, labels) = (data.to(device), labels.to(device))
                outputs = model(data)
                _,pred = torch.max(outputs, 1)
                correct_pred_val += torch.sum(pred==labels)
            
            val_acc = (correct_pred_val/len(val_dataset))*100
            print("Accuracy for the validation set = {}". format(val_acc))

            if val_acc > best_acc:
                print("Saving the best model so far at ~/NNproject/saved_models/resnet{}_model.pth".format(resnet_layer_num))
                torch.save(model.state_dict(),"/afs/crc.nd.edu/user/a/abhatta/NNproject/saved_models/resnet{}_model.pth".format(resnet_layer_num))
                best_acc = val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate ResNet-models ")
    parser.add_argument("--layernum", "-n", required = True, help="Number of ResNet Layer")
    args = parser.parse_args()
    
    train_val(args.layernum)
