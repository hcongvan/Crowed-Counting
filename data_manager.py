import torch
from torch.utils.data import DataLoader
import torch.utils.data as D
import torchvision.transforms as TF
import torchvision.datasets as datasets
import h5py
from PIL import Image
import numpy as np
import os
import cv2

H = 720
W = 1280

class DataManager(D.Dataset):
    def __init__(self, root, target_hdf5, mode = 'train',transforms = None,target_transfroms = None):
        self.augmentation = TF.Compose([
                TF.Resize((H,W)),
                TF.TenCrop((int(H/2),int(W/2)))
            ])
        self.mode = mode
        self.transforms = transforms
        if mode == 'train':
            self.target_transfroms = target_transfroms or TF.Compose([
                TF.Resize((int(H/16),int(W/16))),
                TF.ToTensor()
            ])
        else:
            self.target_transfroms = target_transfroms or TF.Compose([
                TF.Resize((int(H/8),int(W/8))),
                TF.ToTensor()
            ])
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
            if self.mode == 'train':
                _img = self.augmentation(img)
                _target = self.augmentation(target)
                inps = tuple(_img)
                labels = tuple(_target)
                if self.transforms is not None:
                    inps = [self.transforms(_img[i]) for i in range(len(_img))]
                if self.target_transfroms is not None:
                    labels = [self.target_transfroms(_target[i])*64 for i in range(len(_target))]
                return inps,labels
            else:
                img = TF.functional.resize(img,(H,W))
                target = TF.functional.resize(target,(H,W))
                if self.transforms is not None:
                    inps = self.transforms(img)
                if self.target_transfroms is not None:
                    labels = self.target_transfroms(target)*64
                imgs = np.array(img.getdata(),dtype=np.uint8).reshape((img.height,img.width,3))
                imgs = cv2.resize(imgs,(W,H))
                return imgs,inps,labels