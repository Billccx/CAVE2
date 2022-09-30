import argparse
import cv2
from calculators.HandsYolo.yolo import YOLO
from utils.result import Result
from calculators.baseCalculator import BaseCalculator

class HandsDectYolo(BaseCalculator):
    def __init__(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument('-n', '--network', default="tiny", choices=["normal", "tiny", "prn", "v4-tiny"],
                             help='Network Type')
        self.ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
        self.ap.add_argument('-s', '--size', default=416, help='Size for yolo')
        self.ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
        self.ap.add_argument('-nh', '--hands', default=-1,
                             help='Total number of hands to be detected per frame (-1 for all)')
        self.args = self.ap.parse_args()

        if self.args.network == "normal":
            print("loading yolo...")
            self.yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
        elif self.args.network == "prn":
            print("loading yolo-tiny-prn...")
            self.yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
        elif self.args.network == "v4-tiny":
            print("loading yolov4-tiny-prn...")
            self.yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
        else:
            print("loading yolo-tiny...")
            self.yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

        self.yolo.size = int(self.args.size)
        self.yolo.confidence = float(self.args.confidence)


    def Process(self,img):
        width, height, inference_time, results = self.yolo.inference(img)
        ret=Result('hands')
        ret.setResult(results)
        return ret

    def Draw(self,img,**kwargs):
        # sort by confidence
        kwargs['result'].result.sort(key=lambda x: x[2])
        # how many hands should be shown
        hand_count = len(kwargs['result'].result)
        # display hands
        for detection in kwargs['result'].result[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        return img
