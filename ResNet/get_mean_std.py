import torch
import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

# Loading the validation data
val_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Validation/',transform=transform)

# Loading the training data
train_dataset = ImageFolder(root='/afs/crc.nd.edu/user/a/abhatta/NNproject/archive/Training/',transform=transform)


# from torch.utils.data import DataLoader

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

