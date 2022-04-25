import torch
import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import os 


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

#Loading the validation data
#val_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Validation/',transform=transform)
val_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/MORPH3/',transform=transform)

# Loading the training data
#train_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Training/',transform=transform)


from torch.utils.data import DataLoader

image_data_loader = DataLoader(
  val_dataset, 
  # batch size is whole datset
  batch_size=len(val_dataset), 
  shuffle=False, 
  num_workers=0)



def mean_std(loader):
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)


# for i in range(1,11):
#   path = '/afs/crc.nd.edu/user/a/abhatta/MLproject/ms1m_v2/extracted/extracted_part{}'.format(str(i))
#   print(path)
#   #image_loader = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Validation/',transform=transform)