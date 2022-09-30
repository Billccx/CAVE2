from calculators.HandsYolo.handsDectYolo import HandsDectYolo
from calculators.FaceYolo.faceDectYolo import FaceDectYolo
from calculators.HandsPose.handsPose import HandsPose
from calculators.HeadHandLine.line3 import Line3
import numpy as np

class Pipeline3:
    def __init__(self):
        self.face = FaceDectYolo()
        self.hands = HandsDectYolo()
        self.pose = HandsPose()
        self.line = Line3()
        self.calculators={'face':self.face,'hands':self.hands,'pose':self.pose,'line':self.line}
        self.results={}

    def forward(self,img,**kwargs):

        screen = np.zeros(shape=(720, 720, 3))

        for key,calculator in self.calculators.items():
            if(key=='face' or key=='hands'):
                result=calculator.Process(img)
                self.results[key]=result
            elif(key=='pose'):
                result=calculator.Process(img,handsresult=self.results['hands'])
                self.results[key] = result
            elif(key=='line'):
                result = calculator.Process(img,
                                            depthframe=kwargs['depthframe'],
                                            intrinsics=kwargs['intrinsics'],
                                            faceresult=self.results['face'],
                                            handposeresult=self.results['pose'])
                self.results[key] = result


        for key,calculator in self.calculators.items():
            if(key=='line'):
                calculator.Draw(screen,result=self.results[key])
            else:
                calculator.Draw(img,result=self.results[key])

        self.results.clear()
        return img,screen
