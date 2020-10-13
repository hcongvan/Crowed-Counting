import os
import cv2
import numpy as np
import torch
import h5py
import argparse
import scipy.ndimage as scimg

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', required=True, type=str, help='path to folder dataset')
parser.add_argument('-o','--out', default='dataset.hdf5', type=str, help='path to folder dataset')
args = parser.parse_args()

dataset = h5py.File(args.out,'w')
density = dataset.create_group('density')
for roots, dirs, files in os.walk(args.path):
    masks = None
    for i,f in enumerate(files):
        filename,ext = os.path.splitext(f)
        if (ext in ['.jpg','.jpeg'])and(os.path.exists(os.path.join(roots,filename+'.txt'))):
            img = cv2.imread(os.path.join(roots,f))
            # H,W,C = img.shape
            # rH = 720/H
            # rW = 1280/W
            # img = cv2.resize(img,(1280,720))
            print("process file {} | {}/{}".format(f,i,len(files)))
            with open(os.path.join(roots,filename+'.txt')) as f_label:
                data = f_label.read().split('\n')
            pos = []
            if len(data)>0:
                for s in data:
                    if s != '':
                        coords = s.split(' ')
                        pos.append((int(coords[0]),int(coords[1])))
            mask = np.zeros((img.shape[:2]))
            if len(pos)>0:
                for idx,(x,y) in enumerate(pos):
                    _mask = np.zeros((img.shape[:2]))
                    # x = int(x*rW)
                    # y = int(y*rH)
                    _mask[y,x] = 1
                    _pos = pos.copy()
                    del _pos[idx]
                    data = torch.Tensor(_pos)
                    test = torch.Tensor([[x,y]])
                    dist = torch.norm(data - test,dim=1,p=None)
                    k = dist.topk(3,largest=False)
                    sigma = k.values.mean().item()
                    # _mask = scimg.filters.gaussian_filter(_mask,sigma,mode='constant')
                    _mask = cv2.GaussianBlur(_mask,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_CONSTANT) # same performance with scipy.ndimg
                    mask += _mask

            image1 = None
            # mask = cv2.GaussianBlur(mask,(5,5),sigmaX=4,borderType=cv2.BORDER_CONSTANT)
            image1 = cv2.normalize(mask, image1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img2 = cv2.applyColorMap(image1,cv2.COLORMAP_JET)
            # cv2.imshow(filename,img2)
            # cv2.waitKey(0)
            # cv2.destroyWindow(filename)
            mask = np.expand_dims(mask,axis=0)
            density.create_dataset(filename,data=mask)
            
