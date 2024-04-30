import os
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .base_detector import BaseDetector
from ..utils import *

class ReorgFunction(Function):
    @staticmethod
    def forward(ctx, x, stride=2):
        ctx.stride = stride
        
        b, c, h, w = x.size()
        out_h, out_w, out_c = h // stride, w // stride, c * (stride ** 2)
        
        output = x.new_zeros(b, out_c, out_h, out_w)
        
        for i in range(out_c):
            out_channel = i * (stride ** 2)
            x_channel = out_channel // (stride ** 2)
            
            out_idx = 0
            for j in range(0, out_h, stride):
                for k in range(0, out_w, stride):
                    idx = 0
                    for l in range(stride):
                        for m in range(stride):
                            output[:, out_channel + idx, j + l, k + m] = x[:, x_channel, j * stride + l, k * stride + m]
                            idx += 1
                    out_idx += 1
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        
        b, c, h, w = grad_output.size()
        in_h, in_w, in_c = h * stride, w * stride, c // (stride ** 2)
        
        grad_input = grad_output.new_zeros(b, in_c, in_h, in_w)
        
        for i in range(in_c):
            out_channel = i * (stride ** 2)
            
            for j in range(0, in_h, stride):
                for k in range(0, in_w, stride):
                    for l in range(stride):
                        for m in range(stride):
                            grad_input[:, i, j + l, k + m] = grad_output[:, out_channel + l * stride + m, j // stride, k // stride]
        
        return grad_input, None

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        return ReorgFunction.apply(x, self.stride)

class YOLOv2(BaseDetector):
    def __init__(self, 
            classes,
            backbone='',
            image_size=(416, 416),
            anchors=np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=float), 
            coord_scale=1.0, 
            obj_scale=5.0, 
            class_scale=1.0, 
            saturation=1.5,
            jitter=0.3, 
            hue=0.1, 
            flip_prob=0.5
        ):
        super(YOLOv2, self).__init__(classes, backbone)

        self.name = 'YOLOv2'
        self.image_size = image_size
        self.anchors = torch.Tensor(anchors)
        self.num_anchors = len(anchors)
        self.coord_scale = coord_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.saturation = saturation
        self.jitter = jitter 
        self.hue = hue
        self.flip_prob = flip_prob
        
        if self.backbone != '':
            self._model_w_backbone()
        else:
            self._model_wo_backbone()

    def _model_w_backbone(self):
        backbone = self.supported_backbones[self.backbone](pretrained=True)
        backbone, out_channels = keep_convs_only(backbone)
        backbone.add_module('conv', nn.Conv2d(out_channels, 1024, kernel_size=1))
        backbone.add_module('adaptive_avg_pool', nn.AdaptiveAvgPool2d((self.image_size[0] // 32, self.image_size[1] // 32)))

        for param in backbone.parameters():
            param.requires_grad = False

        # linear
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.global_average_pool = nn.AvgPool2d((1, 1))

    def _model_wo_backbone(self):
        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

        # darknet
        self.conv1s, c1 = self._make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = self._make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = self._make_layers(c2, net_cfgs[6])

        stride = 2
        # stride*stride times the channels of conv1s
        self.reorg = ReorgLayer(stride=2)
        # cat [conv1s, conv3]
        self.conv4, c4 = self._make_layers((c1*(stride*stride) + c3), net_cfgs[7])

        # linear
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(c4, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.global_average_pool = nn.AvgPool2d((1, 1))

    def _make_layers(in_channels, net_cfg):
        layers = []

        if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
            for sub_cfg in net_cfg:
                layer, in_channels = self._make_layers(in_channels, sub_cfg)
                layers.append(layer)
        else:
            for item in net_cfg:
                if item == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    out_channels, ksize = item
                    layers.append(nn.Conv2d(in_channels, out_channels, ksize, 1, 1, bias=False))
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.LeakyReLU(0.1, inplace=True))
                    in_channels = out_channels

        return nn.Sequential(*layers), in_channels
        
    def forward(self, image):
        assert image.shape[1:] == (3, self.image_size[0], self.image_size[1]), 'Invalid image shape'

        if self.backbone != '':
            mainnet = self.backbone(image)
        else:
            conv1s = self.conv1s(image)
            conv2 = self.conv2(conv1s)
            conv3 = self.conv3(conv2)
            conv1s_reorg = self.reorg(conv1s)
            cat_1_3 = torch.cat([conv1s_reorg, conv3], dim=1)
            mainnet = self.conv4(cat_1_3)

        conv5 = self.conv5(mainnet)
        global_average_pool = self.global_average_pool(conv5)
        
        # Reshape for output
        batch_size, _, _, _ = global_average_pool.size()
        global_average_pool_reshaped = global_average_pool.view(batch_size, -1, self.num_anchors, self.num_classes + 5)
        
        # Apply activations to predictions
        xy_pred = F.sigmoid(global_average_pool_reshaped[..., 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[..., 2:4])
        iou_pred = F.sigmoid(global_average_pool_reshaped[..., 4:5])
        score_pred = global_average_pool_reshaped[..., 5:]
        
        return torch.cat([xy_pred, wh_pred, iou_pred, score_pred], dim=-1)
        
    def loss(self, prediction, label):
        assert prediction.shape == label.shape, 'Prediction and label shapes do not match'

        assignments, max_overlaps = _anchor_assignment(label['boxes'], self.anchors)
        
        # Reshape prediction and label tensors to
        # (batch_size, grid_size, grid_size, num_anchors, num_classes + 5)
        batch_size, grid_size = prediction.size(0), prediction.size(1)
        prediction = prediction.view(batch_size, grid_size, grid_size, self.num_anchors, -1)
        label = label.view(batch_size, grid_size, grid_size, self.num_anchors, -1)
        
        # Extract components from prediction and label tensors
        xy_pred, wh_pred, iou_pred, score_pred = prediction[..., 0:2], prediction[..., 2:4], prediction[..., 4:5], prediction[..., 5:]
        xy_gt, wh_gt, iou_gt, score_gt = label[..., 0:2], label[..., 2:4], label[..., 4:5], label[..., 5:]
        
        # Assign ground truth values to the corresponding anchors
        xy_gt = xy_gt[..., assignments]
        wh_gt = wh_gt[..., assignments]
        iou_gt = iou_gt[..., assignments]
        score_gt = score_gt[..., assignments]

        # coord loss
        xy_loss = torch.sum((xy_pred - xy_gt)**2 * max_overlaps.unsqueeze(-1))
        wh_loss = torch.sum((wh_pred - wh_gt)**2 * max_overlaps.unsqueeze(-1))

        # conf loss
        iou_loss = torch.sum((iou_pred - iou_gt)**2 * max_overlaps.unsqueeze(-1))
        
        # class loss
        cls_loss = F.binary_cross_entropy_with_logits(score_pred, score_gt)
        
        # Total loss
        total_loss = xy_loss * self.coord_scale\
                + wh_loss * self.coord_scale\
                + iou_loss * self.obj_scale\
                + cls_loss * self.class_scale\
        
        return total_loss / batch_size
    
    def _anchor_assignment(boxes, anchors):
        intersections = torch.min(
            boxes[:, None, 2:],
            (anchors[:, 0] * anchors[:, 1])[:, None]
        ) - torch.max(
            boxes[:, None, :2],
            torch.tensor(0.0, device=boxes.device)
        )
        intersections = torch.clamp(intersections, min=0)
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        anchor_areas = anchors[:, 0] * anchors[:, 1]
        unions = box_areas[:, None] + anchor_areas - intersections[:, :]
        overlaps = intersections[:, :] / unions

        max_overlaps, assignments = overlaps.max(dim=1)

        return assignments, max_overlaps

    def fit(self, 
            dataset,
            epochs,
            batch_size=8,
            optimizer=None,
            scheduler=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            save_dir='checkpoints',
            tensorboard=False,
        ):
        print(f'{self.name} training started...')

        # Save directory
        backbone_name = self.backbone if self.backbone != '' else 'no_backbone'
        dataset_name = dataset.name + dataset.year
        output_dir = os.path.join(save_dir, f'{self.name}_{backbone_name}_{dataset_name}')
        os.makedirs(output_dir, exist_ok=True)

        dataloader = DataLoader(dataset,  batch_size=batch_size,  shuffle=True, collate_fn=collate_fn)

        # Model
        self.train()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

        if not optimizer:
            optimizer = Adam(params=self.parameters(), lr=0.001, betas=(0.9, 0.999))

        if tensorboard:
            writer = SummaryWriter(output_dir)
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for i, (image, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
                # For YOLO models
                targets = self._bbox_to_yolo_format(targets, image_size=self.image_size, C=self.num_classes)

                # put data to device
                image = image.to(device)
                targets = targets.to(device)

                # clear grad for each iteration
                optimizer.zero_grad()

                # forward
                prediction = self(image)
                loss = self.loss(prediction, targets)
                total_loss += loss.item()

                # update
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

            # log
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
            if tensorboard:
                writer.add_scalar('Loss/train', total_loss/len(dataloader), epoch)

        if tensorboard:
            writer.close()

        # save
        path = os.path.join(output_dir, 'model.pth')
        torch.save(self.state_dict(), path)

        print(f'{self.name} training completed. Model saved at ', path)

    def run(self, image):
        '''
        Output shape: list<dictionary<
            'boxes': torch.Tensor, shape (N, 4)
            'confidences': torch.Tensor, shape (N,)
            'labels': torch.Tensor, shape (N,)
        >>
        '''
        assert image.shape[1:] == (3, self.image_size[0], self.image_size[1]), 'Invalid image shape'
        
        prediction = self(image)

        # convert YOLO tensor to bbox
        bbox = self._yolo_to_bbox_format(prediction, image_size=self.image_size, C=self.num_classes)
        
        # nms
        detection_result = []
        for i in range(len(bbox)):
            keep = nms(bbox[i]['boxes'], bbox[i]['confidences'])
            detection_result.append({
                'boxes': bbox[i]['boxes'][keep],
                'confidences': bbox[i]['confidences'][keep],
                'labels': bbox[i]['labels'][keep]
            })

        return detection_result

    def _bbox_to_yolo_format(self, bbox, image_size=(416, 416), S=13, num_anchors=5, C=20):
        """
        Convert bbox format target to YOLO format for YOLOv2.

        Parameters:
        - bbox: list of dictionaries<str, torch.Tensor>
            - 'boxes': Shape (N, 4) # (xmin, ymin, xmax, ymax)
            - 'confidences': Shape (N,)
            - 'labels': Shape (N,)
        - image_size: tuple, (image_width, image_height)
        - S: the number of grid cells along each dimension
        - num_anchors: the number of anchors per grid cell
        - C: the number of classes

        Returns:
        - yolo: YOLO format target, a tensor of shape (batch_size, S, S, num_anchors*(5+C))
            - each 5-tuple is (x, y, w, h, confidence). 
            - (x, y) is the center of the bounding box relative to the grid cell.
            - (w, h) is the width and height of the bounding box relative to the image size.
            - The confidence is 1 if there is an object in the cell, 0 otherwise.
            - The last C elements are the one-hot encoding of the class.
        """
        batch_size = len(bbox)
        yolo = torch.zeros((batch_size, S, S, num_anchors*(5+C)))
        width, height = image_size
        cell_size_x = width / float(S)
        cell_size_y = height / float(S)

        for batch_idx in range(batch_size):
            boxes = bbox[batch_idx]['boxes']
            confidences = bbox[batch_idx]['confidences']
            labels = bbox[batch_idx]['labels']

            for box, confidence, label in zip(boxes, confidences, labels):
                xmin, ymin, xmax, ymax = box
                x, y, w, h = (xmin+xmax)/2.0, (ymin+ymax)/2.0, xmax-xmin, ymax-ymin
                i, j = int(y / cell_size_y), int(x / cell_size_x)
                x_cell, y_cell = x / cell_size_x - j, y / cell_size_y - i
                w_cell, h_cell = w / width, h / height
                for anchor_idx in range(num_anchors):
                    if yolo[batch_idx, i, j, anchor_idx*(5+C)+4] == 0:
                        yolo[batch_idx, i, j, anchor_idx*(5+C):anchor_idx*(5+C)+2] = torch.tensor([x_cell, y_cell])
                        yolo[batch_idx, i, j, anchor_idx*(5+C)+2:anchor_idx*(5+C)+4] = torch.tensor([w_cell, h_cell])
                        yolo[batch_idx, i, j, anchor_idx*(5+C)+4] = confidence
                        break
                yolo[batch_idx, i, j, num_anchors*(5+C)+int(label)] = 1

        return yolo

    def _yolo_to_bbox_format(self, yolo, image_size=(416, 416), S=13, num_anchors=5, C=20):
        """
        Convert YOLO format target to bbox format for YOLOv2.

        Parameters:
        - yolo: YOLO format target, a tensor of shape (batch_size, S, S, num_anchors*(5+C))
        - image_size: tuple, (image_width, image_height)
        - S: the number of grid cells along each dimension
        - num_anchors: the number of anchors per grid cell
        - C: the number of classes

        Returns:
        - bbox: list of dictionaries<str, torch.Tensor>
            - 'boxes': Shape (N, 4) # (xmin, ymin, xmax, ymax)
            - 'confidences': Shape (N,)
            - 'labels': Shape (N,)
        """
        batch_size = yolo.shape[0]
        width, height = image_size
        cell_size_x = width / float(S)
        cell_size_y = height / float(S)

        bbox = []
        for batch_idx in range(batch_size):
            boxes = []
            confidences = []
            labels = []
            for i in range(S):
                for j in range(S):
                    for anchor_idx in range(num_anchors):
                        if yolo[batch_idx, i, j, anchor_idx*(5+C)+4] > 0:
                            x_cell, y_cell, w_cell, h_cell = yolo[batch_idx, i, j, anchor_idx*(5+C):anchor_idx*(5+C)+4]
                            x = (j + x_cell) * cell_size_x
                            y = (i + y_cell) * cell_size_y
                            w = w_cell * width
                            h = h_cell * height
                            xmin = x - w / 2
                            ymin = y - h / 2
                            xmax = x + w / 2
                            ymax = y + h / 2
                            boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
                            confidences.append(yolo[batch_idx, i, j, anchor_idx*(5+C)+4])
                            labels.append(torch.argmax(yolo[batch_idx, i, j, num_anchors*(5+C):num_anchors*(5+C)+C]))

            if len(boxes) > 0:
                bbox.append({'boxes': torch.stack(boxes), 'confidences':torch.tensor(confidences), 'labels': torch.tensor(labels)})

        return bbox