import mediapipe as mp
import cv2
import numpy as np
import os
from utils.result import Result

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

    def Process(self,results,img):
        for item in results:

            if hasattr(item,'multi_hand_landmarks'):
                if item.multi_hand_landmarks:
                    for hand_landmarks in item.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )


            if hasattr(item,'detections'):
                if item.detections:
                    for detection in item.detections:
                        self.mp_drawing.draw_detection(img, detection)


            if hasattr(item,'pose_landmarks'):
                self.mp_drawing.draw_landmarks(
                    img,
                    item.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            if isinstance(item,Result) and item.getInfo()=='hands':
                # sort by confidence
                item.result.sort(key=lambda x: x[2])

                # how many hands should be shown
                hand_count = len(item.result)


                # display hands
                for detection in item.result[:hand_count]:
                    id, name, confidence, x, y, w, h = detection
                    cx = x + (w / 2)
                    cy = y + (h / 2)

                    # draw a bounding box rectangle and label on the image
                    color = (0, 255, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "%s (%s)" % (name, round(confidence, 2))
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

            if isinstance(item,Result) and item.getInfo()=='face':
                for box in item.result:
                    left=box['left']
                    top = box['top']
                    right = box['left'] + box['width']
                    bottom = box['top'] + box['height']
                    landmark=box['landmark']
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), thickness=2)
                    for i in range(5):
                        cv2.circle(img, (landmark[i * 2], landmark[i * 2 + 1]), 1, (0, 255, 0), thickness=-1)

        return img


