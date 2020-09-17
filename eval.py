import torch
import argparse
import numpy as np
import datetime
from CSRNet import CSRNet
import torchvision.transforms as TF
from PIL import Image

parser = argparse.ArgumentParser('CSRNet training tool')
parser.add_argument('-in','--input',required=True,type=str,help="image input")
parser.add_argument('-cfg','--model',required=True,default='config.json',type=str,help="path to cfg model CSRNET")
parser.add_argument('--use_pretrain',default=False,action="store_true",help="using pretrain VGG16 for frontend backbone")
parser.add_argument('--density',required=True,type=str,help="Path to file density hdf5 file")
parser.add_argument('--cuda',default=False,action="store_true",help="set flag to use cpu or gpu")
parser.add_argument('--checkpoint',default=False,action="store_true",help="continue train from checkpoint")
parser.add_argument('-i','--espisode',default=50,type=int,help="max iteration to train")
parser.add_argument('-lr','--learning_rate',default=0.01,type=float,help="learning rate coefficient")
parser.add_argument('-s','--save_point',default=10,type=int,help="define iteration to save checkpoint")
parser.add_argument('-l','--log_path',default='./logs',type=str,help="define logs path to save checkpoint, performace train, parameters train")
parser.add_argument('-bs','--batchsize',default=3,type=int,help="define number of batch size dataset")
parser.add_argument('-wk','--worker',default=3,type=int,help="define number of worker to train")
parser.add_argument('-kck','--keep_checkpoint',default=10,type=int,help="define number of checkpoint can store")
args = parser.parse_args()


def eval(args,model,inp,transfrom,device):
    model.eval()
    
    if args.cuda:
        inps = inps.to(device)
    
    y_pred = model(inps)
    out = transfrom(y_pred)
    return out
    
            

if __name__ == "__main__":
    model = CSRNet(args.model,use_pretrain=args.use_pretrain)
    inp_transform = TF.Compose([
        TF.ToTensor()
    ])
    val_transform = TF.Compose([
        TF.ToPILImage(),
        TF.Resize(224)
    ])

    img = Image.open(args.input)
    img = inp_transform(Image)
    if args.cuda:
        device = torch.device('cuda')
        model.cuda()
        inp = img.to(device)
    else:
        device = torch.device('cpu')

    with open('{}/checkpoints.txt') as f:
        path_checkpoint = f.read()
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['csrnet'])
    out = eval(args,model,inp,val_transform,device)