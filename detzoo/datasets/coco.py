import os
import json
import torch
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torch.utils.data import ConcatDataset
import torchvision.transforms as T

class COCODataset(Dataset):
    def __init__(self, 
                root, 
                year='2017', 
                image_set='train', 
                download=False, 
                transform=T.Compose(
                    [T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]), 
                classes=[]):

        if year not in ["2017"]:
            raise ValueError("Year of COCO dataset must be '2017'")
        if image_set not in ["train", "val", "trainval"]:
            raise ValueError("Image set must be 'train', 'val', or 'trainval'")

        self.name = 'COCO'
        self.root = root
        self.year = year
        self.image_path = os.path.join(root, image_set + year)
        self.annotation_path = os.path.join(root, "annotations")

        # Resources for downloading the COCO dataset
        resources = {
            '2017': {
                'train': ('http://images.cocodataset.org/zips/train2017.zip', '442b8da7639aecaf257c1dceb8ba8c80'),
                'val': ('http://images.cocodataset.org/zips/val2017.zip', 'cced6f7f71b7629ddf16f17bbcfab6b2'),
                'annotations': ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', '8551ee4bb5860311e79dace7e79cb91e')
            }
        }

        if download:
            self.download(resources)
        else:
            assert self._check_exists(), "Dataset not found. You can use download=True to download it"

        self.transform = transform
        root_train = os.path.join(root, "train" + year)
        root_val = os.path.join(root, "val" + year)
        annFile_train = os.path.join(root, "annotations", "instances_train" + year + ".json")
        annFile_val = os.path.join(root, "annotations", "instances_val" + year + ".json")
        annFile_combined = os.path.join(root, "annotations", "instances_trainval" + year + ".json")

        if image_set == 'train':
            self.dataset = CocoDetection(root_train, annFile_train)
            self.coco = COCO(annFile_train)
        elif image_set == 'val':
            self.dataset = CocoDetection(root_val, annFile_val)
            self.coco = COCO(annFile_val)
        elif image_set == 'trainval':
            dataset_train = CocoDetection(root_train, annFile_train)
            dataset_val = CocoDetection(root_val, annFile_val)
            self.dataset = ConcatDataset([dataset_train, dataset_val])

            if os.path.exists(annFile_combined):
                self.coco = COCO(annFile_combined)
            else:
                # Load the annotation files
                with open(annFile_train, 'r') as f:
                    annotations_train = json.load(f)
                with open(annFile_val, 'r') as f:
                    annotations_val = json.load(f)

                # Combine the annotations
                annotations_combined = annotations_train
                annotations_combined['images'].extend(annotations_val['images'])
                annotations_combined['annotations'].extend(annotations_val['annotations'])

                # Save the combined annotations to a new file
                annFile_combined = os.path.join(root, "annotations", "instances_trainval" + year + ".json")
                with open(annFile_combined, 'w') as f:
                    json.dump(annotations_combined, f)

                # Create a new COCO object with the combined annotations
                self.coco = COCO(annFile_combined)

        self.coco_labels = {1: 'person',2: 'bicycle',3: 'car',4: 'motorcycle',5: 'airplane',6: 'bus',
        7: 'train',8: 'truck',9: 'boat',10: 'traffic light',11: 'fire hydrant',13: 'stop sign',14: 'parking meter',
        15: 'bench',16: 'bird',17: 'cat',18: 'dog',19: 'horse',20: 'sheep',21: 'cow',22: 'elephant',23: 'bear',
        24: 'zebra',25: 'giraffe',27: 'backpack',28: 'umbrella',31: 'handbag',32: 'tie',33: 'suitcase',
        34: 'frisbee',35: 'skis',36: 'snowboard',37: 'sports ball',38: 'kite',39: 'baseball bat',40: 'baseball glove',
        41: 'skateboard',42: 'surfboard',43: 'tennis racket',44: 'bottle',46: 'wine glass',47: 'cup',48: 'fork',
        49: 'knife',50: 'spoon',51: 'bowl',52: 'banana',53: 'apple',54: 'sandwich',55: 'orange',56: 'broccoli',
        57: 'carrot',58: 'hot dog',59: 'pizza',60: 'donut',61: 'cake',62: 'chair',63: 'couch',64: 'potted plant',
        65: 'bed',67: 'dining table',70: 'toilet',72: 'tv',73: 'laptop',74: 'mouse',75: 'remote',76: 'keyboard',
        77: 'cell phone',78: 'microwave',79: 'oven',80: 'toaster',81: 'sink',82: 'refrigerator',83: 'book',84: 'clock',
        85: 'vase',86: 'scissors',87: 'teddy bear',88: 'hair drier',89: 'toothbrush',
        }

        self.coco_classes = list(self.coco_labels.values())

        self.classes = classes

        # If no classes are provided, use the default COCO classes
        if len(classes) == 0:
            self.classes = self.coco_classes

    def download(self, resources):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        for key in ['train', 'val', 'annotations']:
            url, md5 = resources[self.year][key]
            download_and_extract_archive(url, download_root=self.root, md5=md5)

    def _check_exists(self):
        # Check if dataset, images and annotations directories exist
        return os.path.exists(self.image_path) and os.path.exists(self.annotation_path)

    def __getitem__(self, i):
        """
        Return value format:
        (   
            <PIL Image>, 
            {
                'boxes': <torch.Tensor>, # Shape (N, 4) (xmin, ymin, xmax, ymax)
                'confidences': <torch.Tensor>, # Shape (N,)
                'labels': <torch.Tensor> # Shape (N,)
            }
        )
        """
        
        img, target = self.dataset[i]
        boxes = []
        labels = []

        width_orig, height_orig = img.size
        img = self.transform(img)
        width_trans, height_trans = img.shape[2], img.shape[1]

        for obj in target:
            name = self.coco_labels[obj['category_id']]
            if name not in self.classes:
                continue
            label = self.classes.index(name)

            bbox = obj['bbox']
            box = [bbox[0] * width_trans / width_orig, 
                bbox[1] * height_trans / height_orig,
                (bbox[0] + bbox[2]) * width_trans / width_orig,
                (bbox[1] + bbox[3]) * height_trans / height_orig]

            labels.append(label)
            boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        confidences = torch.ones(len(labels), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        targets = {}
        targets["boxes"] = boxes
        targets["confidences"] = confidences
        targets["labels"] = labels
        return img, targets

    def __len__(self):
        return len(self.dataset)
