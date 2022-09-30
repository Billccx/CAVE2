import os
import cv2
import argparse
import numpy as np
import utils.result as uresult
from calculators.baseCalculator import BaseCalculator
import copy
import time
import calculators.HandsYoloCuda.handdet as handdet
from cameras import Cameras

class HandsDetectCuda(BaseCalculator):
    def __init__(self):
        pass

    def Process(self,img,**kwargs):
        det = handdet.detect(img)
        result=uresult.Result('handscuda')
        result.setResult(det)
        return result

    def Draw(self,img,**kwargs):
        result=kwargs['result']
        img = handdet.visualize(img, result.result)
        return img
