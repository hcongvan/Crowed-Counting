import numpy as np
import cv2 as cv2
import json

class ROI:
    def __init__(self):
        self.rois = []
        self.ratioH = 1.0
        self.ratioW = 1.0
        
    def drawRoI(self,frame,roi,color,console=True):
        for i in range(1,len(roi)):
            cv2.line(frame, tuple(roi[i-1]), tuple(roi[i]), color, 6)
            if console:
                print(' - Drawed line ' + str(i))
        if len(roi) > 0:
            cv2.line(frame, tuple(roi[-1]), tuple(roi[0]), color, 6)
            if console:
                print(' - Drawed line ' + str(len(roi)))
        return frame

    def drawROIs(self,frame,rois):
        for roi in rois:
            color = tuple([int(np.random.choice(range(256), size=1)) for i in range(3)])
            frame = self.drawRoI(frame,roi,color)
        return frame

    @property
    def getROIs(self):
        return self.rois

    def removeOutBox(self,frame,rois):
        mask = np.zeros(tuple(frame.shape[:2]))
        for roi in rois:
            cv2.fillPoly(mask, np.array([roi], dtype=np.int32), (1))
        frame[mask == 0 ] = [0,0,0]
        return frame