import argparse
import cv2
from calculators.baseCalculator import BaseCalculator
from utils.result import Result
import torch
import numpy as np
import os
from calculators.HandsPose.models.resnet import resnet18,resnet34,resnet50,resnet101
from calculators.HandsPose.models.squeezenet import squeezenet1_1,squeezenet1_0
from calculators.HandsPose.models.shufflenetv2 import ShuffleNetV2
from calculators.HandsPose.models.shufflenet import ShuffleNet
from calculators.HandsPose.models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from calculators.HandsPose.models.rexnetv1 import ReXNetV1
from calculators.HandsPose.hand_data_iter.datasets import draw_bd_handpose

class HandsPose(BaseCalculator):
    def __init__(self):
        self.cnt=0
        self.output=None
        self.pts_hand=None

        self.parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
        self.parser.add_argument('--model_path',
                                 type=str,
                                 #default='D:/CCX/Pipeline/calculators/HandsPose/weights/resnet_50-size-256-wingloss102-0.119.pth',
                                 default='/home/cuichenxi/code/Python/CAVE2/calculators/HandsPose/weights/resnet_50-size-256-wingloss102-0.119.pth',
                                 help='model_path')  # 模型路径
        self.parser.add_argument('--model', type=str, default='resnet_50',
                            help='''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
                    shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''')  # 模型类型

        self.parser.add_argument('--num_classes', type=int, default=42,
                            help='num_classes')  # 手部21关键点， (x,y)*2 = 42
        self.parser.add_argument('--GPUS', type=str, default='0',
                            help='GPUS')  # GPU选择
        self.parser.add_argument('--test_path', type=str, default='./image/',
                            help='test_path')  # 测试图片路径
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256),
                            help='img_size')  # 输入模型图片尺寸
        self.parser.add_argument('--vis', type=bool, default=True,
                            help='vis')  # 是否可视化图片

        self.ops = self.parser.parse_args()  # 解析添加参数

        unparsed = vars(self.ops)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
        for key in unparsed.keys():
            print('{} : {}'.format(key, unparsed[key]))

        # ---------------------------------------------------------------------------
        os.environ['CUDA_VISIBLE_DEVICES'] = self.ops.GPUS

        test_path = self.ops.test_path  # 测试图片文件夹路径

        # ---------------------------------------------------------------- 构建模型
        print('use model : %s' % (self.ops.model))

        if self.ops.model == 'resnet_50':
            model_ = resnet50(num_classes=self.ops.num_classes, img_size=self.ops.img_size[0])
        elif self.ops.model == 'resnet_18':
            model_ = resnet18(num_classes=self.ops.num_classes, img_size=self.ops.img_size[0])
        elif self.ops.model == 'resnet_34':
            model_ = resnet34(num_classes=self.ops.num_classes, img_size=self.ops.img_size[0])
        elif self.ops.model == 'resnet_101':
            model_ = resnet101(num_classes=self.ops.num_classes, img_size=self.ops.img_size[0])
        elif self.ops.model == "squeezenet1_0":
            model_ = squeezenet1_0(num_classes=self.ops.num_classes)
        elif self.ops.model == "squeezenet1_1":
            model_ = squeezenet1_1(num_classes=self.ops.num_classes)
        elif self.ops.model == "shufflenetv2":
            model_ = ShuffleNetV2(ratio=1., num_classes=self.ops.num_classes)
        elif self.ops.model == "shufflenet_v2_x1_5":
            model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=self.ops.num_classes)
        elif self.ops.model == "shufflenet_v2_x1_0":
            model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=self.ops.num_classes)
        elif self.ops.model == "shufflenet_v2_x2_0":
            model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=self.ops.num_classes)
        elif self.ops.model == "shufflenet":
            model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=self.ops.num_classes, groups=3)
        elif self.ops.model == "mobilenetv2":
            model_ = MobileNetV2(num_classes=self.ops.num_classes)
        elif self.ops.model == "ReXNetV1":
            model_ = ReXNetV1(num_classes=self.num_classes)

        self.use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model_ = model_.to(device)
        self.model_.eval()  # 设置为前向推断模式

        # 加载测试模型
        if os.access(self.ops.model_path, os.F_OK):  # checkpoint
            chkpt = torch.load(self.ops.model_path, map_location=device)
            self.model_.load_state_dict(chkpt)
            print('load test model : {}'.format(self.ops.model_path))
        else:
            print("failed to load")


    def Process(self,img,**kwargs):
        self.pts_hand = None
        result=Result('pose')
        with torch.no_grad():
            handsresult=kwargs['handsresult']
            img_ = img.copy()
            x1 = 0
            y1 = 0

            if len(handsresult.result):
                sorted(handsresult.result, key=lambda x: x['bbox'][1])
                box = handsresult.result[0]['bbox']
                x_min, y_min = box[0], box[1]
                x_max, y_max = box[2], box[3]

                w_ = max(abs(x_max - x_min), abs(y_max - y_min))
                w_ = w_ * 1.1

                x_mid = (x_max + x_min) / 2
                y_mid = (y_max + y_min) / 2

                x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

                x1 = np.clip(x1, 0, img.shape[1] - 1)
                x2 = np.clip(x2, 0, img.shape[1] - 1)

                y1 = np.clip(y1, 0, img.shape[0] - 1)
                y2 = np.clip(y2, 0, img.shape[0] - 1)

                img_ = img_[y1:y2, x1:x2, :]
                #cv2.imwrite('/home/cuichenxi/code/Python/Pipeline/calculators/HandsPose/crop/'+str(self.cnt)+'.jpg',img_)
                self.cnt+=1

                # cv2.namedWindow('i')
                # cv2.imshow('i', img_)
                # cv2.imwrite('result/'+file,img)

                img_width = img_.shape[1]
                img_height = img_.shape[0]

                # 输入图片预处理
                img_ = cv2.resize(img_, (self.ops.img_size[1], self.ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
                img_ = img_.astype(np.float32)
                img_ = (img_ - 128.) / 256.

                img_ = img_.transpose(2, 0, 1)
                img_ = torch.from_numpy(img_)
                img_ = img_.unsqueeze_(0)

                if self.use_cuda:
                    img_ = img_.cuda()  # (bs, 3, h, w)
                pre_ = self.model_(img_.float())  # 模型推理
                self.output = pre_.cpu().detach().numpy()
                self.output = np.squeeze(self.output)

                self.pts_hand = {}  # 构建关键点连线可视化结构
                for i in range(int(self.output.shape[0] / 2)):
                    x = (self.output[i * 2 + 0] * float(img_width))
                    y = (self.output[i * 2 + 1] * float(img_height))

                    self.pts_hand[str(i)] = {}
                    self.pts_hand[str(i)] = {
                        "x": x + x1,
                        "y": y + y1,
                    }

                result.setResult([self.pts_hand['8']['x'],self.pts_hand['8']['y']])
        #print("POSE RESULT {}".format(result.result))
        return result


    def Draw(self,img,**kwargs):
        if self.pts_hand:
            draw_bd_handpose(img, self.pts_hand, 0, 0)  # 绘制关键点连线





