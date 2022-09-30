import mediapipe as mp
import numpy as np
import os
import cv2

class HandsDect:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    def Process(self,img):
        img.flags.writeable = False
        result = self.hands_detector.process(img)
        return result
