from abc import ABCMeta,abstractmethod

class BaseCalculator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def Process(self,img,**kwargs):
        pass

    @abstractmethod
    def Draw(self,img,**kwargs):
        pass

