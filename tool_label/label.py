import cv2
import numpy as np
# from drawROI import DrawROI
from labelDot import labelDot
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', required=True, type=str, help='path to folder dataset')
parser.add_argument('-z','--height', default=None,type=int, help='scale height image')
parser.add_argument('-w','--width', default=None,type=int, help='scale width image')
args = parser.parse_args()

if __name__ == "__main__":
    scale = None
    if (args.height is not None)and(args.width is not None):
        scale = (args.width,args.height)
    for root,dirs,files in os.walk(args.path):
        for f in files:
            if os.path.splitext(f)[-1] in ['.jpeg','.jpg','.png']:
                filename = os.path.join(root,f)
                name = os.path.splitext(filename)[0]
                img = cv2.imread(filename)
                label = labelDot(name +'.txt')
                label.setBackground(img,scale)
                label.label()#(980,720))