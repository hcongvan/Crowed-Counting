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
import uuid
import os
import shutil
import json
from ROI_Settings import ROI

parser = argparse.ArgumentParser('CSRNet training tool')
parser.add_argument('--input',required=True,type=str,help="path to images test")
parser.add_argument('-r','--roi',default='config.json',type=str,help="path to RoI setting")
parser.add_argument('-cfg','--model',required=True,default='config.json',type=str,help="path to cfg model CSRNET")
parser.add_argument('--cuda',default=False,action="store_true",help="set flag to use cpu or gpu")
parser.add_argument('--checkpoint',default=False,action="store_true",help="continue train from checkpoint")
parser.add_argument('-l','--log_path',default='./logs',type=str,help="define logs path to save checkpoint, performace train, parameters train")
args = parser.parse_args()


def eval(args,model,reader,setting,device):
    fps = reader.get(cv2.CAP_PROP_FPS)
    H = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter('./results/{}.mp4'.format(str(uuid.uuid4())), cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H),True)
    model = model.eval()
    transforms = TF.Compose([
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    i = 0
    n = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    while(i < n):
        if args.cuda:
            torch.cuda.empty_cache()
        # reader.set(cv2.CAP_PROP_POS_FRAMES ,i)
        ret, frame = reader.read()
        if ret:
            origin = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image = setting.removeOutBox(origin,args.rois)
            img = Image.fromarray(image)
            inps = transforms(img)
            inps = inps.unsqueeze(0)
            if args.cuda:
                inps = inps.to(device)
            y_pred = model(inps)
            y_pred_tt = y_pred.sum().detach().item()
            

        _img = y_pred.squeeze(0).permute(1,2,0).detach().to(torch.device('cpu')).numpy()
        _img = (_img*255/_img.max()).astype(np.uint8)
        _img = cv2.resize(_img,(W,H))
        _img = cv2.merge([_img,_img,_img])
        denisity = cv2.addWeighted(_img,0.7,origin,0.3,0.3)
        cv2.putText(denisity, 'count:{}'.format(y_pred_tt), (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                               1)
        # cv2.imshow('result',denisity)
        # cv2.waitKey(0)
        writer.write(denisity)
        i += 1
        # print(reader.get(cv2.CAP_PROP_POS_FRAMES))
    writer.release()
    reader.release()
            

if __name__ == "__main__":
    # current_time = datetime.datetime.now().strftime('%m%d%Y-%H%M%S')
    # writer = SummaryWriter(args.log_path+'/CSRnet-eval-{}'.format(current_time))
    model = CSRNet(args.model)
    reader = cv2.VideoCapture(args.input)
    if args.cuda:
        device = torch.device('cuda')
        model.cuda()
    else:
        device = torch.device('cpu')
    with open(args.roi) as f:
        rois = json.load(f).get('settings').get('rois')
    args.rois = rois
    setting = ROI()
    with open('{}/checkpoint.txt'.format(args.log_path)) as f:
        path_checkpoint = f.read().split('\n')[-1]
        checkpoint = torch.load(path_checkpoint,map_location=device)
        model.load_state_dict(checkpoint['csrnet'])
    eval(args,model,reader,setting,device)
