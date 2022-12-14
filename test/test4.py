# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
from cameras import Cameras

detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
cams = Cameras()
cams.captureRGB(0)

while True:
    img = cams.getRGBFrame(0)
    # 摘自官方文档：
    # image is a numpy ndarray containing either an 8bit grayscale or RGB image.
    # opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
    b, g, r = cv2.split(img)  # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片

    dets = detector(img, 1)  # 使用detector进行人脸检测 dets为返回的结果
    print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
    # enumerate是一个Python的内置方法，用于遍历索引
    # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
    # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))

        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.imshow('img', img)
    cv2.waitKey(1)


