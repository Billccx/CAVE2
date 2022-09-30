import pyrealsense2 as rs
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from cameras import Cameras,CameraIntrinsics
from graphs.pipeline5 import Pipeline2


if __name__=='__main__':
    mp_drawing = mp.solutions.drawing_utils
    pipe = Pipeline2()
    cams = Cameras()
    cams.captureRGBandDepth(0)
    cams.captureRGBandDepth(1)

    f_count=0
    t1=time.time()
    while (True):
        if f_count>30:
            t1=time.time()
            f_count=0
        f_count+=1

        img0,depth0=cams.getRGBandDepthFrame(0)
        img1,depth1=cams.getRGBandDepthFrame(1)
        # img1=img0
        # depth1=depth0


        img,screen = pipe.forward(
            img0.colorframe,
            img1.colorframe,
            depthframe0=depth0,
            depthframe1=depth1,
            intrinsics0=img0.intrinsics,
            intrinsics1=img1.intrinsics
        )


        fps = f_count / (time.time() - t1)
        cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)


        cv2.namedWindow("img")
        #cv2.moveWindow('img', 200, 20)
        cv2.imshow('img', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        cv2.namedWindow("screen")
        #cv2.moveWindow('screen', 1300, 500)
        cv2.imshow('screen', screen)

        if cv2.waitKey(5) & 0xFF == 27:
            break
