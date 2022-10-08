from calculators.FaceYoloCuda.faceDetectCuda import FaceDetectCuda
from calculators.HandsYoloCuda.handsDetectCuda import HandsDetectCuda
from calculators.HeadHandLine.lineCuda import LineCuda
from calculators.HandsPose.handsPose import HandsPose
import numpy as np
import cv2

class Pipeline1:
    def __init__(self):
        self.linecuda = LineCuda()
        self.facecuda=FaceDetectCuda()
        self.handscuda=HandsDetectCuda()
        self.handpose=HandsPose()
        self.calculators = {'facecuda': self.facecuda,'handscuda':self.handscuda,'linecuda':self.linecuda}
        self.results={}
        self.usepose=True
        if self.usepose:
            self.calculators = {'facecuda': self.facecuda, 'handscuda': self.handscuda,'handspose':self.handpose, 'linecuda': self.linecuda}
        else:
            self.calculators = {'facecuda': self.facecuda, 'handscuda': self.handscuda, 'linecuda': self.linecuda}

    def forward(self,img,**kwargs):

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
            if(key=='facecuda' or key=='handscuda'):
                result=calculator.Process(img)
                self.results[key]=result
            elif (key == 'handspose'):
                result = calculator.Process(img, handsresult=self.results['handscuda'])
                self.results[key] = result
            elif(key=='linecuda'):
                if not self.usepose:
                    result = calculator.Process(img,
                                                depthframe=kwargs['depthframe'],
                                                intrinsics=kwargs['intrinsics'],
                                                faceresult=self.results['facecuda'],
                                                handsresult=self.results['handscuda'],
                                                usepose=False)
                    self.results[key] = result
                else:
                    result = calculator.Process(img,
                                                depthframe=kwargs['depthframe'],
                                                intrinsics=kwargs['intrinsics'],
                                                faceresult=self.results['facecuda'],
                                                handsresult=self.results['handspose'],
                                                usepose=True)
                    self.results[key] = result



        for key,calculator in self.calculators.items():
            if(key=='linecuda'):
                calculator.Draw(screen,result=self.results[key])
            else:
                calculator.Draw(img,result=self.results[key])

        self.results.clear()
        return img,screen
