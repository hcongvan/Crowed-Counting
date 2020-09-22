import torch
import argparse
import numpy as np
import datetime
from CSRNet import CSRNet
from data_manager import DataManager
from torch.utils.data import DataLoader
import torchvision.transforms as TF
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2

parser = argparse.ArgumentParser('CSRNet training tool')
parser.add_argument('--test',required=True,type=str,help="path to images test")
parser.add_argument('-cfg','--model',required=True,default='config.json',type=str,help="path to cfg model CSRNET")
parser.add_argument('--density',required=True,type=str,help="Path to file density hdf5 file")
parser.add_argument('--cuda',default=False,action="store_true",help="set flag to use cpu or gpu")
parser.add_argument('--checkpoint',default=False,action="store_true",help="continue train from checkpoint")
parser.add_argument('-l','--log_path',default='./logs',type=str,help="define logs path to save checkpoint, performace train, parameters train")
parser.add_argument('-bs','--batchsize',default=3,type=int,help="define number of batch size dataset")
parser.add_argument('-wk','--worker',default=3,type=int,help="define number of worker to train")
args = parser.parse_args()


def eval(args,model,loader,transfrom,device):
    model = model.eval()
    display_loss = np.array([])
    for idx,(inps,labels) in enumerate(loader):
        inps = torch.cat(inps,dim=0)
        labels = torch.cat(labels,dim=0)
        # labels= labels.squeeze(dim=1)
        if args.cuda:
            inps = inps.to(device)
            labels = labels.to(device)
        
        y_pred = model(inps)
        loss = euclidean_dist(y_pred,labels).detach()
        out = y_pred.detach()
        display_loss = np.append(display_loss,loss.item())
        print("loss : {}".format(loss.item()),end='\r')
    print("avg_loss: {}\n".format(display_loss.mean()))
    

    
            

if __name__ == "__main__":
    # current_time = datetime.datetime.now().strftime('%m%d%Y-%H%M%S')
    # writer = SummaryWriter(args.log_path+'/CSRnet-eval-{}'.format(current_time))
    euclidean_dist = torch.nn.MSELoss(reduction='sum')
    model = CSRNet(args.model)
    inp_transform = TF.Compose([
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    val_transform = TF.Compose([
        TF.ToPILImage(),
        TF.Resize((224,224))
    ])

    manager = DataManager(args.test,args.density,inp_transform)
    loader = DataLoader(manager,batch_size=args.batchsize,shuffle=False)
    if args.cuda:
        device = torch.device('cuda')
        model.cuda()
    else:
        device = torch.device('cpu')

    with open('{}/checkpoint.txt'.format(args.log_path)) as f:
        path_checkpoint = f.read().split('\n')[-1]
        checkpoint = torch.load(path_checkpoint,map_location=device)
        model.load_state_dict(checkpoint['csrnet'])
    eval(args,model,loader,val_transform,device)
