import torch
import torch.nn as nn
from torchvision.models import *

class BaseDetector(nn.Module):
    def __init__(self, classes, backbone=''):
        super(BaseDetector, self).__init__()
        self.classes = classes

        # Define the supported backbones
        # Concrete backbone selection is done in the derived class
        self.supported_backbones = {
            'alexnet': alexnet(weights=AlexNet_Weights.DEFAULT),
            'vgg16': vgg16(weights=VGG16_Weights.DEFAULT),
            'vgg19': vgg19(weights=VGG19_Weights.DEFAULT),
            'resnet50': resnet50(weights=ResNet50_Weights.DEFAULT),
            'resnet101': resnet101(weights=ResNet101_Weights.DEFAULT),
        }
        if backbone != '' and backbone not in self.supported_backbones.keys():
            raise ValueError('Invalid backbone')
        self.backbone = backbone

    def _model_w_backbone(self):
        """
        Implementation depends on the specific detector.
        """
        raise NotImplementedError

    def _model_wo_backbone(self):
        """
        Implementation depends on the specific detector.
        """
        raise NotImplementedError

    def forward(self, image):
        """
        Implementation depends on the specific detector.
        """
        raise NotImplementedError

    def loss(self, prediction, label):
        """
        Implementation depends on the specific detector.
        """
        raise NotImplementedError

    def run(self, image):
        """
        Implementation depends on the specific detector.
        """
        raise NotImplementedError
