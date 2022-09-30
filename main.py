import pyrealsense2 as rs
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from cameras import Cameras,CameraIntrinsics
from graphs.pipeline1 import Pipeline1
from calculators.HandsYolo.handsDectYolo import HandsDectYolo




if __name__=='__main__':
    mp_drawing = mp.solutions.drawing_utils
    pipe = Pipeline1()
    cams = Cameras()
    cams.captureRGBandDepth(0)
    #cams.captureRGB(0)
    f_count=0
    t1=time.time()
    while (True):
        if f_count>30:
            t1=time.time()
            f_count=0
        f_count+=1
        #img = cams.getRGBFrame(0)
        img,depth=cams.getRGBandDepthFrame(0)

        img,screen = pipe.forward(img.colorframe,depthframe=depth,intrinsics=img.intrinsics)

        fps = f_count / (time.time() - t1)
        cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)

        cv2.imshow('img', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        cv2.namedWindow("screen")
        cv2.moveWindow('screen', 1700,500)
        cv2.imshow('screen', screen)

        if cv2.waitKey(5) & 0xFF == 27:
            break



    # mp_drawing = mp.solutions.drawing_utils
    # cams=Cameras()
    # cams.captureRGB(0)
    # #pipe=Pipeline1()
    #
    # while (True):
    #     img = cams.getRGBFrame(0)
    #     #img=pipe.forward(img)
    #
    #     cv2.imshow(img)
    #     cv2.waitKey(0)
