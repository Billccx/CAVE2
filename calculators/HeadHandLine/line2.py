import math

import cv2
import numpy as np
import pyrealsense2 as rs2
from calculators.baseCalculator import BaseCalculator
from utils.result import Result

class Line2(BaseCalculator):
    def __init__(self,path):
        f=open('/home/cuichenxi/code/Python/extrinsics2/'+path,'r',encoding='utf-8')
        R_ = []
        t_ = []
        knt = 0
        for line in f:
            row = []
            line = line.strip()
            line = line.split(' ')
            row.append(float(line[0]))
            row.append(float(line[1]))
            row.append(float(line[2]))
            if (knt < 3):
                R_.append(row)
            else:
                t_.append(row)
            knt += 1

        self.R = np.array(R_).T
        self.t=np.array(t_).reshape(3, 1)

    def Process(self,img,**kwargs):
        '''
        :param img: colorframe
        :param kwargs: depthframe, intrinsics, faceresult, handsresult
        :return:
        '''
        depth=kwargs['depthframe']
        intrinsics=kwargs['intrinsics']
        faceresult=kwargs['faceresult']
        handsresult=kwargs['handsresult']

        hasHead=False
        hasHand=False

        headpoint3d = [0, 0, 0]
        if len(faceresult.result):

            headlandmarks=faceresult.result[0]['landmark']
            #两眼中心
            headpoint2d = (
                (headlandmarks[0] + headlandmarks[2]) / 2,
                (headlandmarks[1] + headlandmarks[3]) / 2
            )

            if int(headpoint2d[0])>=0 and int(headpoint2d[0])<640 and int(headpoint2d[1])>=0 and int(headpoint2d[1])<480: #change more elegant way
                headpoint_depth = depth.get_distance(int(headpoint2d[0]), int(headpoint2d[1]))
                headpoint3d = rs2.rs2_deproject_pixel_to_point(intrin=intrinsics, pixel=headpoint2d,depth=headpoint_depth)
                hasHead = True
                #print("head: {}".format(headpoint3d))



        handpoint3d = [0, 0, 0]
        if len(handsresult.result):
            handslandmark=handsresult.result[0]
            id, name, confidence, x, y, w, h = handslandmark
            handpoint2d=(x+w/2,y+h/2)
            handpoint_depth = depth.get_distance(int(handpoint2d[0]),int(handpoint2d[1]))
            handpoint3d=rs2.rs2_deproject_pixel_to_point(intrin=intrinsics,pixel=handpoint2d,depth=handpoint_depth)
            hasHand = True
            #print("hand: {}".format(handpoint3d))

        headpoint3d = np.array(headpoint3d).reshape(3, 1)*1000
        handpoint3d = np.array(handpoint3d).reshape(3, 1)*1000
        headpoint3d_trans=np.matmul(self.R,headpoint3d)+self.t
        handpoint3d_trans=np.matmul(self.R,handpoint3d)+self.t
        print("\n\nhead\n",headpoint3d_trans,'\nhand\n',handpoint3d_trans)

        x=0
        y=0
        if hasHand and hasHead:
            x = headpoint3d_trans[0][0] - headpoint3d_trans[2][0] / (headpoint3d_trans[2][0] - handpoint3d_trans[2][0]) * (headpoint3d_trans[0][0] - handpoint3d_trans[0][0])
            y = headpoint3d_trans[1][0] - headpoint3d_trans[2][0] / (headpoint3d_trans[2][0] - handpoint3d_trans[2][0]) * (headpoint3d_trans[1][0] - handpoint3d_trans[1][0])
            x = -x


        result=Result('line')
        result.setResult((x,y))

        return result





    def Draw(self,img,**kwargs):
        x = kwargs['result'].result[0] / 0.1797 / 8904 * 500
        y = kwargs['result'].result[1] / 0.1815 / 8815 * 500
        print("from cam {}, coordinate ({},{})".format(kwargs['index'],x,y))

        if math.isnan(x) or x==float('inf') or x==float('-inf'):
            x=0
        if math.isnan(y) or y==float('inf') or y==float('-inf'):
            y=0

        if (kwargs['index'] == 0):
            cv2.circle(img=img, center=(int(x), int(y)), radius=30, color=(0, 0, 255), thickness=cv2.FILLED)
        else:
            cv2.circle(img=img, center=(int(x), int(y)), radius=30, color=(0, 255, 0), thickness=cv2.FILLED)



