import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 初始化一个图像数组
img = np.zeros(shape=(480, 640,3))

cv2.circle(img,(100,200),5,(0,0,255))

cv2.imshow("img", img)
cv2.waitKey(0)