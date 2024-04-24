import torch
import torch.nn as nn
from torchvision.models import *
from torch.optim import Adam

class BaseDetector(nn.Module):
    def __init__(self, classes, backbone=''):
        super(BaseDetector, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)

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
            optimizer=Adam(params=self.parameters(), lr=0.001, betas=(0.9, 0.999)),
            scheduler=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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
