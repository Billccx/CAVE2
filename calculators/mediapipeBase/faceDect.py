import mediapipe as mp
import cv2
import os
import numpy as np

class FaceDect:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def Process(self,img):
        img.flags.writeable = False
        results = self.face_detector.process(img)
        return results
