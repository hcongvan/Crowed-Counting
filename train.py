import torch
from torch.utils.data import DataLoader
import argparse
from CSRNet import CSRNet
from data_manager import DataManager
import torchvision.transforms as TF

parser = argparse.ArgumentParser('CSRNet training tool')
args = parser.parse_args()
path = '/home/vanhc/dataset/vov_dataset/train'
density_path = path + '/dataset.hdf5'

model = CSRNet('model.json',use_pretrain=True)
opt = torch.optim.Adam(model.parameters(),lr=0.01)
loss = torch.nn.CrossEntropyLoss()
target_transforms = TF.Compose([
    TF.Resize(28),
    TF.ToTensor()
])
inp_transform = TF.ToTensor()
manager = DataManager(path,density_path,inp_transform,target_transforms)
loader = DataLoader(manager,batch_size=3,shuffle=False,num_workers=3)
cuda = False
device = torch.device('cuda')
if __name__ == "__main__":
    for i in range(50):
        for idx,(inps,labels) in enumerate(loader):
            inps = torch.cat(inps,dim=0)
            labels = torch.cat(labels,dim=0)

            if cuda:
                inps.to(device)
                labels.to(device)
                model.to(device)
            
            y_pred = model(inps)
            loss = loss(y_pred,labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

