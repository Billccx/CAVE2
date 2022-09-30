import calculators.FaceYolo.yolov5 as v5
import cv2
import argparse
import numpy as np
import utils.result
from calculators.baseCalculator import BaseCalculator

class FaceDectYolo(BaseCalculator):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--yolo_type', type=str, default='yolov5s', choices=['yolov5s', 'yolov5m', 'yolov5l'],
                            help="yolo type")
        self.parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
        self.parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
        self.parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
        self.args = self.parser.parse_args()
        self.yolonet = v5.yolov5(self.args.yolo_type, confThreshold=self.args.confThreshold, nmsThreshold=self.args.nmsThreshold,
                         objThreshold=self.args.objThreshold)
        self.inpWidth = 640
        self.inpHeight = 640


    def PostProcess(self,dets,img):
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth

        confidences = []
        boxes = []
        landmarks = []
        for detection in dets:
            confidence = detection[15]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.args.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = detection[5:15] * np.tile(np.float32([ratiow, ratioh]), 5)
                landmarks.append(landmark.astype(np.int32))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.args.confThreshold, self.args.nmsThreshold)
        result=utils.result.Result('face')
        result.setResult([])


        for i in indices:
            box = boxes[i]
            boxinfo={}
            boxinfo['left']=box[0]
            boxinfo['top'] = box[1]
            boxinfo['width'] = box[2]
            boxinfo['height'] = box[3]
            boxinfo['landmark'] = landmarks[i]
            result.result.append(boxinfo)

        return result


    def Process(self,img,**kwargs):
        dets = self.yolonet.detect(img)
        result= self.PostProcess(dets,img)
        return result

    def Draw(self,img,**kwargs):
        for box in kwargs['result'].result:
            left = box['left']
            top = box['top']
            right = box['left'] + box['width']
            bottom = box['top'] + box['height']
            landmark = box['landmark']
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(img, (landmark[i * 2], landmark[i * 2 + 1]), 1, (0, 255, 0), thickness=-1)
        return img

