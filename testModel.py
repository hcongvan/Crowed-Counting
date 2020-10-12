import torch
import numpy as np
import torch.utils.data.DataLoader as DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

tf = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.TenCrop((224,224)),
    transforms.ToTensor()])

inp = datasets.ImageFolder('/home/vanhc/Projects/AI_Dataset/Dataset/VOVTraffic/',transform=tf)
tar =