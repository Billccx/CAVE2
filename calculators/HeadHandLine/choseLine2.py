from calculators.baseCalculator import BaseCalculator
import cv2
import os
import math

class ChoseLine(BaseCalculator):
    def __init__(self):
        pass
    def Process(self,img,**kwargs):
        '''
        :param img:
        :param kwargs:
        :return:
        '''
        pass
    def Draw(self,img,**kwargs):
        '''
        :param img:
        :param kwargs:
        :return:
        '''
        point0=kwargs['pt0'].result
        point1=kwargs['pt1'].result
        result=None
        index=0
        if not point0[2]:
            result=point0
        else:
            print('use right')
            index=1
            result=point1

        # x = result[0] / 0.1797 / (1085/0.1797) * 1070
        # y = result[1] / 0.1815 / (730/0.1815) * 720
        x = result[0] / 0.233
        y = result[1] / 0.233
        #print("in choose Line, coordinate ({},{})".format(x, y))

        canShow = True

        if (x == 0 and y == 0):
            canShow = False

        if math.isnan(x) or x == float('inf') or x == float('-inf'):
            canShow = False
        if math.isnan(y) or y == float('inf') or y == float('-inf'):
            canShow = False

        if canShow:
            if (index == 0):
                cv2.circle(img=img, center=(int(x), int(y)), radius=30, color=(255, 0, 255), thickness=cv2.FILLED)
                print("in chose Line, (0, 0, 255) coordinate ({},{})".format(x, y))
                pass
            else:
                #cv2.circle(img=img, center=(int(x), int(y)), radius=30, color=(0, 255, 0), thickness=cv2.FILLED)
                pass

