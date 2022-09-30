import cv2
from calculators.FaceYolo.faceDectYolo import FaceDectYolo
from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.mediapipeBase.visualizer import Visualizer
import time


cap=cv2.VideoCapture('basicvideo2.mp4')
hands=HandsDectYolo()
faces=FaceDectYolo()
vis=Visualizer()
cnt=0
f_count=0
t1=time.time()
while(cap.isOpened()):
    if f_count > 30:
        t1 = time.time()
        f_count = 0
    f_count += 1
    cnt += 1
    success, img = cap.read()


    result0 = hands.Process(img)
    result1 = faces.Process(img)
    frame0 = vis.Process([result0,result1], img)


    fps = f_count / (time.time() - t1)
    cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(5)
