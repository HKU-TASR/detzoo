import os
from tqdm import tqdm
import torch
from torch import nn
from . import BaseDetector
from ..utils import *

class FasterRCNN(BaseDetector):
    def __init__(self,
            classes,
            backbone='vgg16',
            image_size=800,
            num_anchor=9,
            roi_pooling_size=(7, 7),
        ):
        super(FasterRCNN, self).__init__(classes, backbone)

        self.image_size = image_size

        # (N, image_size, image_size, 3)

        # Feature extractor
        backbone = self.supported_backbones[backbone]
        backbone, out_channels = keep_convs_only(backbone)
        backbone.add_module('conv', nn.Conv2d(out_channels, 512, kernel_size=1))
        self.feature_extractor = backbone

        # (N, H, W, 512)

        # RPN
        self.rpn_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.rpn_loc_layer = nn.Conv2d(512, num_anchor * 4, kernel_size=1) # (N, H, W, num_anchor * 4)
        self.rpn_cls_layer = nn.Conv2d(512, num_anchor * 2, kernel_size=1) # (N, H, W, num_anchor * 2)

        # ROI Classifier
        self.roi_pool = nn.AdaptiveMaxPool2d(roi_pooling_size)

        # (N, 7, 7, 512)

        input_dim = 512 * roi_pooling_size[0] * roi_pooling_size[1]
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(input_dim, 4096),
                                                   nn.ReLU(),
                                                   nn.Linear(4096, 4096),
                                                   nn.ReLU()])
        # (N, 4096)
        self.roi_loc_layer = nn.Linear(4096, (self.num_classes + 1) * 4) # (N, (num_classes + 1) * 4)
        self.roi_cls_layer = nn.Linear(4096, self.num_classes + 1) # (N, num_classes + 1)

    def forward(self, image):
        '''
        Image pass through RPN.
        '''
        assert image.size[1:] == (3, self.image_size, self.image_size), 'Invalid image shape'
        
        feature_map = self.feature_extractor(image)

        rpn_conv_feature = self.rpn_conv(feature_map)
        pred_anchor_locs = self.rpn_loc_layer(rpn_conv_feature)
        pred_cls_scores = self.rpn_cls_layer(rpn_conv_feature)

        return feature_map, pred_anchor_locs, pred_cls_scores

    def loss(self, prediction, label):
        '''
        Faster R-CNN use rpn_loss and roi_loss instead.
        '''
        pass

    def rpn_loss(self, rpn_loc, rpn_cls, gt_rpn_loc, gt_rpn_cls, weight=10.0):
        gt_rpn_cls = torch.autograd.Variable(gt_rpn_cls.long())
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_cls, gt_rpn_cls, ignore_index=-1)

        pos = gt_rpn_cls.data > 0
        mask = pos.unsqueeze(1).expand_as(rpn_loc)

        mask_pred_loc = rpn_loc[mask].view(-1, 4)
        mask_target_loc = gt_rpn_loc[mask].view(-1, 4)

        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        rpn_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))
        rpn_loc_loss = rpn_loc_loss.sum() 

        N_reg = (gt_rpn_cls > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())

        rpn_loc_loss = rpn_loc_loss / N_reg
        rpn_loc_loss = np.float32(rpn_loc_loss)

        rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
        rpn_loss = rpn_cls_loss + (weight * rpn_loc_loss)
        return rpn_loss

    def roi_loss(self, pre_loc, pre_conf, target_loc, target_conf, weight=10.0):
        target_conf = torch.autograd.Variable(target_conf.long())
        pred_conf_loss = torch.nn.functional.cross_entropy(pre_conf, target_conf, ignore_index=-1)

        pos = target_conf.data > 0
        mask = pos.unsqueeze(1).expand_as(pre_loc)

        mask_pred_loc = pre_loc[mask].view(-1, 4)
        mask_target_loc = target_loc[mask].view(-1, 4)

        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())

        pre_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))

        N_reg = (target_conf > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())
        pre_loc_loss = pre_loc_loss.sum() / N_reg
        pre_loc_loss = np.float32(pre_loc_loss)
        pred_conf_loss = np.squeeze(pred_conf_loss.data.numpy())
        total_loss = pred_conf_loss + (weight * pre_loc_loss)

        return total_loss

    def fit(self,
            train_loader,
            epochs,
            optimizer=torch.optim.Adam(params=self.parameters(), lr=0.001, betas=(0.9, 0.999)),
            scheduler=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            save_dir='checkpoints',
        ):
        pass

    def run(self, image):
        pass