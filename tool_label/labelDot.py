import numpy as np
import cv2 as cv2
import json
import os

class labelDot:
    def __init__(self, configfile):
        self.config = []
        data = []
        self.configfile = configfile
        if os.path.exists(configfile):
            with open(configfile,'r') as f:
                data = f.read().split('\n')
        if len(data)>0:
            for s in data:
                if s != '':
                    coords = s.split(' ')
                    self.config.append([int(j) for j in coords])
        self.points = self.config or []
        self.ratioH = 1.0
        self.ratioW = 1.0

    def setBackground(self, frame,scale):
        self.background = frame
        self.H,self.W,self.C = frame.shape
        if scale is not None:
            self.background = cv2.resize(self.background,scale)
            self.ratioH = scale[1]/self.H
            self.ratioW = scale[0]/self.W
            if len(self.points)>0:
                for idx,(x,y) in enumerate(self.points):
                    self.points[idx][0] = int(x*self.ratioW)
                    self.points[idx][1] = int(y*self.ratioH)

    def label(self):
        # Create a black image, a window and bind the function to window
        img = self.background.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)
        # Load exist configuration data
        img = self.draw_points(img)
        flag_inspect = True
        while (1):
            cv2.imshow('image', img)
            if flag_inspect:
                img = self.draw_points(img)
            key = cv2.waitKey(20)
            if key & 0xFF == ord('q'):
                self.points = self.config
                break
            elif key & 0xFF == ord('s'):
                with open(self.configfile,'w') as f:
                    for i in self.points:
                        f.write('{} {}\n'.format(int(i[0]/self.ratioW),int(i[1]/self.ratioH)))
                print('@ Saved ROI lines configuration.')
            elif key & 0xFF == ord('z'):
                if len(self.points) > 0:
                    img = self.background.copy()
                    del self.points[-1]
                    img = self.draw_points(img)
                    print('=> Cleared point')
            elif key & 0xFF == ord('c'):
                img = self.background.copy()
                self.points.clear()
                print('=> Cleared all point')
        cv2.destroyAllWindows()

    # mouse callback function
    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x,y))
            
    def draw_points(self,frame):
        for i in self.points:
            x,y = i
            cv2.circle(frame,(x,y),2,(255,0,0),-1)
        return frame
        