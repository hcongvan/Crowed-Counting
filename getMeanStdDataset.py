import torch
import torchvision.datasets as dataset
import torchvision.transforms as TF

path = '/home/vanhc/Projects/dataset/vov_dataset'
tf = TF.Compose([
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
dsets = dataset.ImageFolder(path,tf)
for i,x in dsets:
    pass
print("TEST")