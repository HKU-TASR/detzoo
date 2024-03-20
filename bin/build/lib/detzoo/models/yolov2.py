import torch
from torch import nn
from . import BaseDetector
from ..utils import iou, Reshape, PrintShape, keep_convs_only

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class YOLOv2(BaseDetector):
    def __init__(self, 
            classes, 
            backbone='', 
            image_size = 416,
            grid_size=13,
            anchor_num=5, 
        ):
        super(YOLOv2, self).__init__(classes, backbone)

        self.image_size = image_size
        self.S = grid_size
        self.B = anchor_num
        self.num_classes = len(self.classes)
        self.D = anchor_num * 5 + self.num_classes

        self.conv1 = nn.Sequential(
            Conv_BN_ReLU(3, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv_BN_ReLU(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv_BN_ReLU(64, 128, 3, 1),
            Conv_BN_ReLU(128, 64, 1, 1),
            Conv_BN_ReLU(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv_BN_ReLU(128, 256, 3, 1),
            Conv_BN_ReLU(256, 128, 1, 1),
            Conv_BN_ReLU(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv_BN_ReLU(256, 512, 3, 1),
            Conv_BN_ReLU(512, 256, 1, 1),
            Conv_BN_ReLU(256, 512, 3, 1),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv_BN_ReLU(512, 1024, 3, 1),
            Conv_BN_ReLU(1024, 512, 1, 1),
            Conv_BN_ReLU(512, 1024, 3, 1),
            Conv_BN_ReLU(1024, 512, 1, 1),
            Conv_BN_ReLU(512, 1024, 3, 1),
        )

        self.conv3 = nn.Sequential(
            Conv_BN_ReLU(1024, 1024, 3, 1),
            Conv_BN_ReLU(1024, 1024, 3, 1),
        )

        self.conv4 = nn.Sequential(
            Conv_BN_ReLU(1024, 1024, 3, 1),
        )


    def forward(self, image):
        return self.model(image)

    def loss(self, prediction, label):
        pass
