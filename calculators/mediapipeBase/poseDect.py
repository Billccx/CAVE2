import mediapipe as mp
import numpy as np
import os
import cv2

class PoseDect:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def Process(self,img):
        img.flags.writeable = False
        results = self.pose_detector.process(img)
        return results