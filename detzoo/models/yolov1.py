import os
from tqdm import tqdm
import torch
from torch import nn
from . import BaseDetector
from ..utils import *

class YOLOv1(BaseDetector):
    def __init__(self, 
            classes, 
            backbone='', 
            image_size = 448,
            S=7, # Number of cells in the grid
            B=2, # Number of bounding boxes per cell
            lambda_coord=5,
            lambda_noobj=0.5,
        ):
        super(YOLOv1, self).__init__(classes, backbone)

        self.image_size = image_size
        self.S = S
        self.B = B
        self.D = B * 5 + self.num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        if self.backbone != '':
            self.model = self._model_w_backbone()
        else:
            self.model = self._model_wo_backbone()

    def _model_w_backbone(self):
        backbone = self.supported_backbones[self.backbone]
        backbone, out_channels = keep_convs_only(backbone)
        backbone.add_module('conv', nn.Conv2d(out_channels, 2048, kernel_size=1))
        backbone.add_module('adaptive_avg_pool', nn.AdaptiveAvgPool2d((2 * self.S, 2 * self.S)))

        model = nn.Sequential(
                # PrintShape("Input"), # (N, 3, 448, 448)
                backbone,
                # PrintShape("Backbone"), # (N, 2048, 14, 14)
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                # PrintShape("Conv2d"), # (N, 1024, 7, 7)
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Flatten(),
                # PrintShape("Flatten"), # (N, 7 * 7 * 1024)
                nn.Linear(7 * 7 * 1024, 4096),
                # PrintShape("Linear"), # (N, 4096)
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(4096, self.S * self.S * self.D),
                # PrintShape("Linear"), # (N, 7 * 7 * 30)
                Reshape(-1, self.S, self.S, self.D),
                # PrintShape("Output") # (N, 7, 7, 30)
            )

        return model

    def _model_wo_backbone(self):
        layers = [
                # PrintShape("Input"), # (N, 3, 448, 448)
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),                   # Conv 1
                # PrintShape("Conv2d"), # (N, 64, 224, 224)
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # PrintShape("MaxPool2d"), # (N, 64, 112, 112)
                nn.Conv2d(64, 192, kernel_size=3, padding=1),                           # Conv 2
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # PrintShape("MaxPool2d"), # (N, 192, 56, 56)
                nn.Conv2d(192, 128, kernel_size=1),                                     # Conv 3
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # PrintShape("MaxPool2d"), # (N, 512, 28, 28)
            ]

        for _ in range(4):                                                          # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # PrintShape("MaxPool2d"), # (N, 1024, 14, 14)
        ]

        for _ in range(2):                                                          # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            # PrintShape("Conv2d"), # (N, 1024, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),
        ]

        for _ in range(2):                                                          # Conv 6
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        layers += [
            nn.Flatten(),
            # PrintShape("Flatten"), # (N, 7 * 7 * 1024)
            nn.Linear(self.S * self.S * 1024, 4096),                            # Linear 1
            # PrintShape("Linear"), # (N, 4096)
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, self.S * self.S * self.D),                      # Linear 2
            # PrintShape("Linear") # (N, 7 * 7 * 30)
        ]

        layers.append(Reshape(-1, self.S, self.S, self.D))
        # PrintShape("Output") # (N, 7, 7, 30)

        return nn.Sequential(*layers)

    def forward(self, image):
        '''
        Output shape: (N, S, S, D)
        '''
        assert image.shape[1:] == (3, self.image_size, self.image_size), 'Invalid image shape'
        return self.model(image)

    def loss(self, prediction, label):
        assert prediction.shape == label.shape, 'Prediction and label shapes do not match'

        # input shape: (N, S, S, D)
        # coordinates: (xmin, ymin, xmax, ymax, c)
        b_size = prediction.shape[0]

        # Split the prediction and label tensor into its components
        # shape: (N, S, S, B, 5)
        pred_boxes = prediction[..., :5 * self.B].view(b_size, self.S, self.S, self.B, 5)
        label_boxes = label[..., :5 * self.B].view(b_size, self.S, self.S, self.B, 5)
        # shape: (N, S, S, num_classes)
        pred_class = prediction[..., 5 * self.B:]
        label_class = label[..., 5 * self.B:]

        # Compute the IoU for each predicted bounding box
        # shape: (N, S, S, B)
        ious = torch.stack([iou(pred_boxes[..., b, :4], label_boxes[..., b, :4]) for b in range(self.B)], dim=-1)

        # Only consider the bounding box with the highest IoU for the coordinate and confidence loss
        # shape: (N, S, S)
        iou_maxes, iou_max_indices = ious.max(dim=-1)

        # shape: (N, S, S, 5)
        pred_boxes_best = pred_boxes.gather(-2, iou_max_indices.unsqueeze(-1).unsqueeze(-1).expand_as(pred_boxes[..., :1, :])).squeeze(-2)
        label_boxes_best = label_boxes.gather(-2, iou_max_indices.unsqueeze(-1).unsqueeze(-1).expand_as(label_boxes[..., :1, :])).squeeze(-2)

        # Create masks for the cells that contain an object and the cells that do not contain an object
        obj_mask = label_boxes_best[..., 4] > 0
        # shape: (N, S, S, 1)
        obj_mask = obj_mask.unsqueeze(-1).expand_as(pred_boxes_best[..., :1])
        noobj_mask = ~obj_mask

        # Compute the mean squared error for the coordinates
        coord_loss = nn.MSELoss()(pred_boxes_best[..., :2] * obj_mask, label_boxes_best[..., :2] * obj_mask)

        # Compute the width and height
        pred_width_height = pred_boxes_best[..., 2:4] - pred_boxes_best[..., :2]
        label_width_height = label_boxes_best[..., 2:4] - label_boxes_best[..., :2]

        # Compute the dimension loss
        dim_loss = nn.MSELoss()(torch.sqrt(pred_width_height * obj_mask), torch.sqrt(label_width_height * obj_mask))

        # Compute the mean squared error for the confidence
        # shape: (N, S, S)
        obj_mask = obj_mask.squeeze(-1)
        noobj_mask = noobj_mask.squeeze(-1)
        obj_conf_loss = nn.MSELoss()(pred_boxes_best[..., 4] * obj_mask, iou_maxes * obj_mask)
        noobj_conf_loss = nn.MSELoss()(pred_boxes_best[..., 4] * noobj_mask, iou_maxes * noobj_mask)

        # Compute the cross entropy loss for the class predictions
        class_loss = nn.CrossEntropyLoss()(pred_class.view(-1, self.num_classes), torch.argmax(label_class, dim=-1).view(-1))

        # Combine the losses
        total_loss = self.lambda_coord * coord_loss \
                    + self.lambda_coord * dim_loss \
                    + obj_conf_loss \
                    + self.lambda_noobj * noobj_conf_loss \
                    + class_loss

        return total_loss / b_size

    def fit(self, 
            train_loader, 
            epochs,
            optimizer=Adam(params=self.parameters(), lr=0.001, betas=(0.9, 0.999)),
            scheduler=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            save_dir='checkpoints',
        ):
        print('YOLOv1 training started...')

        self.train()
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)
        
        for epoch in tqdm(range(epochs), desc='Epoch'):
            for image, targets in tqdm(train_dataloader, desc='Train', leave=False):
                # For YOLO models
                targets = self._bbox_to_yolo_format(targets)

                # put data to device
                image = image.to(device)
                targets = targets.to(device)

                # clear grad for each iteration
                self.optimizer.zero_grad()

                # forward
                prediction = self(image)
                loss = self.loss(prediction, targets)

                # update
                loss.backward()
                self.optimizer.step()

                if scheduler:
                    scheduler.step()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, f'YOLOv1_{backbone}_{dataset}.pth')
        torch.save(self.state_dict(), path)

        print('YOLOv1 training completed. Model saved at ', path)

    def run(self, image):
        '''
        Output shape: list<dictionary<
            'boxes': torch.Tensor, shape (N, 4)
            'confidences': torch.Tensor, shape (N,)
            'labels': torch.Tensor, shape (N,)
        >>
        '''
        assert image.shape[1:] == (3, self.image_size, self.image_size), 'Invalid image shape'
        
        prediction = self(image)

        # convert YOLO tensor to bbox
        bbox = self._yolo_to_bbox_format(prediction)
        
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

    def _bbox_to_yolo_format(self, bbox, image_size=(448, 448), S=7, B=2, C=20):
        """
        Convert bbox format target to YOLO format for YOLOv1.

        Parameters:
        - bbox: list of dictionaries<str, torch.Tensor>
            - 'boxes': Shape (N, 4) # (xmin, ymin, xmax, ymax)
            - 'confidences': Shape (N,)
            - 'labels': Shape (N,)
        - image_size: tuple, (image_width, image_height)
        - S: the number of grid cells along each dimension
        - B: the number of bounding boxes per grid cell
        - C: the number of classes. For VOC, C=20. For COCO, C=80.

        Returns:
        - yolo: YOLO format target, a tensor of shape (batch_size, S, S, B*5+C)
            - each 5-tuple is (x, y, w, h, confidence). 
            - (x, y) is the center of the bounding box relative to the grid cell.
            - (w, h) is the width and height of the bounding box relative to the image size.
            - The confidence is 1 if there is an object in the cell, 0 otherwise.
            - The last C elements are the one-hot encoding of the class.
        """
        batch_size = len(bbox)
        yolo = torch.zeros((batch_size, S, S, B*5+C))
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
                i, j = int(y / cell_size_x), int(x / cell_size_y)
                x_cell, y_cell = x / cell_size_x - j, y / cell_size_y - i
                w_cell, h_cell = w / width, h / height
                for b in range(B):
                    if yolo[batch_idx, i, j, b*5+4] == 0:
                        yolo[batch_idx, i, j, b*5:b*5+2] = torch.tensor([x_cell, y_cell])
                        yolo[batch_idx, i, j, b*5+2:b*5+4] = torch.tensor([w_cell, h_cell])
                        yolo[batch_idx, i, j, b*5+4] = confidence
                        break
                yolo[batch_idx, i, j, B*5+int(label)] = 1

        return yolo

    def _yolo_to_bbox_format(self, yolo, image_size=(448, 448), S=7, B=2, C=20):
        """
        Convert YOLO format target to bbox format.

        Parameters:
        - yolo: YOLO format target, a tensor of shape (batch_size, S, S, B*5+C)
        - image_size: tuple, (image_width, image_height)
        - S: the number of grid cells along each dimension
        - B: the number of bounding boxes per grid cell
        - C: the number of classes. For VOC, C=20. For COCO, C=80.

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
                    for b in range(B):
                        if yolo[batch_idx, i, j, b*5+4] > 0:
                            x_cell, y_cell, w_cell, h_cell = yolo[batch_idx, i, j, b*5:b*5+4]
                            x = (j + x_cell) * cell_size_x
                            y = (i + y_cell) * cell_size_y
                            w = w_cell * width
                            h = h_cell * height
                            xmin = x - w / 2
                            ymin = y - h / 2
                            xmax = x + w / 2
                            ymax = y + h / 2
                            boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
                            confidences.append(yolo[batch_idx, i, j, b*5+4])
                            labels.append(torch.argmax(yolo[batch_idx, i, j, B*5:B*5+C]))

            if len(boxes) > 0:
                bbox.append({'boxes': torch.stack(boxes), 'confidences':torch.tensor(confidences), 'labels': torch.tensor(labels)})

        return bbox