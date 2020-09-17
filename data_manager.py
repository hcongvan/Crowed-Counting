import torch
from torch.utils.data import DataLoader
import torch.utils.data as D
import torchvision.transforms as TF
import torchvision.datasets as datasets
import h5py
from PIL import Image
import numpy as np
import os

class DataManager(D.Dataset):
    def __init__(self, root, target_hdf5, transforms = None,target_transfroms = None):
        self.augmentaion = TF.Compose([
            TF.Resize((448,448)),
            TF.TenCrop((224,224))
        ])
        self.transforms = transforms
        self.target_transfroms = target_transfroms
        self.root_path = root
        self.files = os.listdir(root)
        self.target_file = target_hdf5
        with h5py.File(target_hdf5,'r') as f:
            self.n = len(f['density'])
            self.list_key = list(f['density'].keys())

    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        key = self.list_key[idx]
        if not(key+'.jpg' in self.files):
            return None,None
        with h5py.File(self.target_file,'r') as f:
            target = f['density'][key][()]
            target = target.squeeze(0)
            target = Image.fromarray(target.copy())
            img = Image.open(os.path.join(self.root_path,key+'.jpg'))
            _img = self.augmentaion(img)
            _target = self.augmentaion(target)
            inps = tuple(_img)
            labels = tuple(_target)
            if self.transforms is not None:
                inps = [self.transforms(_img[i]) for i in range(len(_img))]
            if self.target_transfroms is not None:
                labels = [self.target_transfroms(_target[i]) for i in range(len(_target))]
            return inps,labels