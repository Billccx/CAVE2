from calculators.FaceYoloCuda.faceDetectCuda import FaceDetectCuda
from calculators.HandsYoloCuda.handsDetectCuda import HandsDetectCuda
from calculators.HeadHandLine.lineCuda2 import LineCuda2
from calculators.HeadHandLine.lineCuda3 import LineCuda
from calculators.HeadHandLine.choseLine2 import ChoseLine
from calculators.HandsPose.handsPose import HandsPose
import numpy as np
import cv2

class Pipeline2:
    def __init__(self):
        self.facecuda0 = FaceDetectCuda()
        self.handscuda0 = HandsDetectCuda()

        self.facecuda1 = FaceDetectCuda()
        self.handscuda1 = HandsDetectCuda()

        self.linecuda0 = LineCuda2('0toscreen.txt')
        self.linecuda1 = LineCuda2('1toscreen.txt')

        self.handspose0 = HandsPose()
        self.handspose1 = HandsPose()

        self.choseline = ChoseLine()

        self.usepose=True

        if not self.usepose:
            self.calculators={
                'face0':self.facecuda0,
                'face1':self.facecuda1,
                'hands0':self.handscuda0,
                'hands1':self.handscuda1,
                'line0':self.linecuda0,
                'line1':self.linecuda1,
                'chose':self.choseline,
            }
        else:
            # self.calculators = {
            #     'face0': self.facecuda0,
            #     'hands0': self.handscuda0,
            #     'pose0':self.handspose0,
            #     'line0': self.linecuda0,
            # }

            self.calculators = {
                'face0': self.facecuda0,
                'face1': self.facecuda1,
                'hands0': self.handscuda0,
                'hands1': self.handscuda1,
                'pose0': self.handspose0,
                'pose1': self.handspose1,
                'line0': self.linecuda0,
                'line1': self.linecuda1,
                'chose': self.choseline,
            }

        self.results={}

    def forward(self,img0,img1,**kwargs):

        screen = np.zeros(shape=(1440, 3440, 3))
        cv2.rectangle(screen, (1000, 200), (1100, 300), (255, 255, 0), 3)
        cv2.rectangle(screen, (2000, 200), (2100, 300), (255, 255, 0), 3)
        cv2.rectangle(screen, (3000, 200), (3100, 300), (255, 255, 0), 3)

        cv2.rectangle(screen, (1000, 700), (1100, 800), (255, 255, 0), 3)
        cv2.rectangle(screen, (2000, 700), (2100, 800), (255, 255, 0), 3)
        cv2.rectangle(screen, (3000, 700), (3100, 800), (255, 255, 0), 3)

        cv2.rectangle(screen, (1000, 1200), (1100, 1300), (255, 255, 0), 3)
        cv2.rectangle(screen, (2000, 1200), (2100, 1300), (255, 255, 0), 3)
        cv2.rectangle(screen, (3000, 1200), (3100, 1300), (255, 255, 0), 3)

        for key,calculator in self.calculators.items():
            if(key=='face0' or key=='hands0'):
                result=calculator.Process(img0)
                self.results[key]=result
            elif (key == 'face1' or key == 'hands1'):
                result = calculator.Process(img1)
                self.results[key] = result
            elif (key == 'pose0'):
                result = calculator.Process(img0, handsresult=self.results['hands0'])
                self.results[key] = result
            elif (key == 'pose1'):
                result = calculator.Process(img1, handsresult=self.results['hands1'])
                self.results[key] = result
            elif (key == 'line0'):
                if not self.usepose:
                    result = calculator.Process(img0,
                                                depthframe=kwargs['depthframe0'],
                                                intrinsics=kwargs['intrinsics0'],
                                                faceresult=self.results['face0'],
                                                handsresult=self.results['hands0'],
                                                usepose=False)
                    self.results[key] = result
                else:
                    result = calculator.Process(img0,
                                                depthframe=kwargs['depthframe0'],
                                                intrinsics=kwargs['intrinsics0'],
                                                faceresult=self.results['face0'],
                                                handsresult=self.results['pose0'],
                                                usepose=True)
                    self.results[key] = result
            elif (key == 'line1'):
                if not self.usepose:
                    result = calculator.Process(img1,
                                                depthframe=kwargs['depthframe1'],
                                                intrinsics=kwargs['intrinsics1'],
                                                faceresult=self.results['face1'],
                                                handsresult=self.results['hands1'],
                                                usepose=False)
                    self.results[key] = result
                else:
                    result = calculator.Process(img1,
                                                depthframe=kwargs['depthframe1'],
                                                intrinsics=kwargs['intrinsics1'],
                                                faceresult=self.results['face1'],
                                                handsresult=self.results['pose1'],
                                                usepose=True)
                    self.results[key] = result

            '''
                elif(key=='line0'):
                    result = calculator.Process(img0,
                                                depthframe=kwargs['depthframe0'],
                                                intrinsics=kwargs['intrinsics0'],
                                                faceresult=self.results['face0'],
                                                handsresult=self.results['hands0'])
                    self.results[key] = result
                elif (key == 'line1'):
                    result = calculator.Process(img1,
                                                depthframe=kwargs['depthframe1'],
                                                intrinsics=kwargs['intrinsics1'],
                                                faceresult=self.results['face1'],
                                                handsresult=self.results['hands1'])
                    self.results[key] = result
            '''


        for key,calculator in self.calculators.items():
            if(key=='chose'):
                    calculator.Draw(screen,pt0=self.results['line0'],pt1=self.results['line1'])
            elif (key=='line0'):
                calculator.Draw(screen,result=self.results[key],index=0)
                pass
            elif (key == 'line1'):
                calculator.Draw(screen, result=self.results[key],index=1)
                pass
            elif(key[-1]=='0'):
                calculator.Draw(img0,result=self.results[key])
            elif (key[-1] == '1'):
                calculator.Draw(img1, result=self.results[key])

        img=cv2.hconcat([img0,img1])

        self.results.clear()
        return img,screen
