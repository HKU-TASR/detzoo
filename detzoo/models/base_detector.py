import torch
import torch.nn as nn
from torchvision.models import *

class BaseDetector(nn.Module):
    def __init__(self, classes, backbone=''):
        super(BaseDetector, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)

        # Define the supported backbones
        # Concrete backbone selection is done in the derived class
        self.supported_backbones = {
            'alexnet': alexnet,
            'densenet121': densenet121,
            'densenet161': densenet161,
            'densenet169': densenet169,
            'densenet201': densenet201,
            'googlenet': googlenet,
            'inceptionv3': inception_v3,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'vgg11': vgg11,
            'vgg11_bn': vgg11_bn,
            'vgg13': vgg13,
            'vgg13_bn': vgg13_bn,
            'vgg16': vgg16,
            'vgg16_bn': vgg16_bn,
            'vgg19': vgg19,
            'vgg19_bn': vgg19_bn,
        }
        self.backbone = backbone

    def forward(self, image):
        """
        Network forward pass.
        """
        raise NotImplementedError

    def loss(self, prediction, label):
        """
        Compute the loss.
        """
        raise NotImplementedError

    def fit(self, 
            train_loader, 
            epochs,
            optimizer=None,
            scheduler=None,
            device='cuda',
            save_dir='checkpoints',
        ):
        """
        Train the model.
        """
        raise NotImplementedError

    def run(self, image):
        """
        Run the model on the input image. For testing stage.
        """
        raise NotImplementedError
