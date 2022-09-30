from calculators.FaceYoloCuda.faceDetectCuda import FaceDetectCuda
from calculators.HandsYoloCuda.handsDetectCuda import HandsDetectCuda
from calculators.HeadHandLine.lineCuda3 import LineCuda
from calculators.HandsPose.handsPose import HandsPose
import numpy as np

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

        screen = np.zeros(shape=(720, 1720, 3))

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
