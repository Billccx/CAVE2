from cameras import Cameras
import cv2
import os
import numpy as np

cams=Cameras()
cams.captureRGBandDepth(0)
while True:
    color,depth=cams.getRGBandDepthFrame(0)
    cv2.imshow('img',color)
    cv2.waitKey(1)