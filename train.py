import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import datetime
from CSRNet import CSRNet
from data_manager import DataManager
import torchvision.transforms as TF
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('CSRNet training tool')
parser.add_argument('--train',required=True,type=str,help="Path to train folder")
parser.add_argument('-cfg','--model',required=True,default='config.json',type=str,help="path to cfg model CSRNET")
parser.add_argument('--use_pretrain',default=False,type=bool,help="using pretrain VGG16 for frontend backbone")
parser.add_argument('--density',required=True,type=str,help="Path to file density hdf5 file")
parser.add_argument('--cuda',default=False,type=bool,help="set flag to use cpu or gpu")
parser.add_argument('--checkpoint',default=False,type=bool,help="continue train from checkpoint")
parser.add_argument('-i','--espisode',default=50,type=int,help="max iteration to train")
parser.add_argument('-lr','--learning_rate',default=0.01,type=float,help="learning rate coefficient")
parser.add_argument('-s','--save_point',default=10,type=int,help="define iteration to save checkpoint")
parser.add_argument('-l','--log_path',default='./logs',type=str,help="define logs path to save checkpoint, performace train, parameters train")
parser.add_argument('-bs','--batchsize',default=3,type=int,help="define number of batch size dataset")
args = parser.parse_args()

writer = SummaryWriter(args.log_path+'/CSRnet')
model = CSRNet(args.model,use_pretrain=True)
opt = torch.optim.Adam(model.parameters(),lr=args.learnig_rate)
euclidean_dist = torch.nn.MSELoss(reduction='sum')
target_transforms = TF.Compose([
    TF.Resize(28),
    TF.ToTensor()
])
inp_transform = TF.ToTensor()
val_transform = TF.Compose([
    TF.ToPILImage(),
    TF.Resize(224)
])
manager = DataManager(args.train,args.density,inp_transform,target_transforms)
loader = DataLoader(manager,batch_size=args.batchsize,shuffle=False,num_workers=3)

if args.cuda:
    device = torch.device('cuda')
    model.cuda()
    euclidean_dist.cuda()
else:
    device = torch.device('cpu')
def train():
    model.train()
    display_loss = np.array([])
    for i in range(args.espisode):
        for idx,(inps,labels) in enumerate(loader):
            inps = torch.cat(inps,dim=0)
            labels = torch.cat(labels,dim=0)
            # labels= labels.squeeze(dim=1)
            if args.cuda:
                inps.to(device)
                labels.to(device)
            
            y_pred = model(inps)
            loss = euclidean_dist(y_pred,labels)
            display_loss = np.append(display_loss,loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        writer.add_scalar('train/loss',display_loss.mean(),global_step=i)
        if i % args.save_point:
            current_time = datetime.datetime.now().strftime('%m%d%Y-%H%M%S')
            torch.save({
                'csrnet':model.state_dict(),
                'opt':opt.state_dict()
            },'logs/checkpoint-{}.pth'.format(current_time))
            with open('logs/checkpoints.txt','w') as f:
                f.write('logs/checkpoint-{}.pth'.format(current_time))

