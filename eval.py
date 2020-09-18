import torch
import argparse
import numpy as np
import datetime
from CSRNet import CSRNet
import torchvision.transforms as TF
from PIL import Image
import cv2

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
    inp = inp.unsqueeze(0)
    if args.cuda:
        inp = inp.to(device)
    
    y_pred = model(inp)
    test1 = y_pred.max()
    y_pred = y_pred.squeeze(0).to(torch.device('cpu'))
    # out = transfrom(y_pred)
    return y_pred
    
            

if __name__ == "__main__":
    model = CSRNet(args.model,use_pretrain=args.use_pretrain)
    inp_transform = TF.Compose([
        TF.Resize((224,224)),
        TF.ToTensor()
    ])
    val_transform = TF.Compose([
        TF.ToPILImage(),
        TF.Resize((224,224))
    ])

    img = Image.open(args.input)
    inp = inp_transform(img)
    if args.cuda:
        device = torch.device('cuda')
        model.cuda()
        inp = inp.to(device)
    else:
        device = torch.device('cpu')

    with open('{}/checkpoints.txt'.format(args.log_path)) as f:
        path_checkpoint = f.read().split('\n')[-1]
        checkpoint = torch.load(path_checkpoint,map_location=device)
        model.load_state_dict(checkpoint['csrnet'])
    out = eval(args,model,inp,val_transform,device)
    tt = out.sum().detach()
    out = out.squeeze(0)
    src2 = np.array(out.detach().numpy())
    src2 = (src2*255/src2.max()).astype(np.uint8)
    src2 = cv2.resize(src2,(224,224))
    # src2 = cv2.merge([src2,src2,src2])
    cv2.imshow('out',src2)
    cv2.waitKey(0)