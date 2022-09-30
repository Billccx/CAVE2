import cv2
import argparse
import numpy as np
import utils.result as uresult
from calculators.baseCalculator import BaseCalculator
import copy
import time
from cameras import Cameras

import torch
from calculators.FaceYoloCuda.models.experimental import attempt_load
from calculators.FaceYoloCuda.utils.datasets import letterbox
from calculators.FaceYoloCuda.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from calculators.FaceYoloCuda.utils.plots import plot_one_box
from calculators.FaceYoloCuda.utils.torch_utils import select_device, load_classifier, time_synchronized


class FaceDetectCuda(BaseCalculator):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='D:/CCX/Pipeline/calculators/FaceYoloCuda/weights/yolov5n-face.pt',
                            help='model.pt path(s)')
        parser.add_argument('--image', type=str, default='data/images/bus.jpg',
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        self.opt = parser.parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model

    def scale_coords_landmarks(self,img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    def detect_one(self,image):
        img_size = 800
        conf_thres = 0.3
        iou_thres = 0.5

        img0 = copy.deepcopy(image)

        h0, w0 = image.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # Process detections
        boxes=[]
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()

                for j in range(det.size()[0]):
                    # 字典box用于存储检测信息
                    box = {}
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    box['pt1'] = (int(xyxy[0]), int(xyxy[1]))
                    box['pt2'] = (int(xyxy[2]), int(xyxy[3]))
                    box['landmarks']=landmarks
                    box['conf']=conf
                    box['class_num']=class_num
                    box['size'] = abs(int(xyxy[0]) - int(xyxy[2])) * abs(int(xyxy[1]) - int(xyxy[3]))
                    boxes.append(box)
                    #image = self.show_results(image, xyxy, conf, landmarks, class_num)

        result = uresult.Result('faceCuda')
        result.setResult(boxes)
        return result

    def show_results(self,img, pt1, pt2, conf, landmarks, class_num):
        h, w, c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

        x1 = pt1[0]
        y1 = pt1[1]
        x2 = pt2[0]
        y2 = pt2[1]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        for i in range(5):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
            cv2.putText(img, str(i), (point_x, point_y), 0, tl / 3, [225, 255, 255], thickness=max(tl - 1, 1), lineType=cv2.LINE_AA)

        tf = max(tl - 1, 1)  # font thickness
        label = str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def Process(self,img,**kwargs):
        result = self.detect_one(img)
        return result

    def Draw(self,img,**kwargs):
        result=kwargs['result']
        for box in result.result:
            self.show_results(img,box['pt1'],box['pt2'],box['conf'],box['landmarks'],box['class_num'])
        return img

