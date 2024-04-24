import os
import torch
import json
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
from PIL import Image
from urllib.error import URLError

class COCODataset(VisionDataset):
    def __init__(self, root, year='2017', train=True, download=False, transform=None):
        super(COCODataset, self).__init__(root, transform=transform)
        if year not in ["2017"]:
            raise ValueError("Year of COCO dataset must be '2017'")
        self.year = year
        self.train = train

        # Paths for the images and annotations
        self.data_folder = os.path.join(root, 'COCO')
        self.image_folder = os.path.join(self.data_folder, '{}{}'.format('train' if train else 'val', year))
        self.annotation_folder = os.path.join(self.data_folder, 'annotations')
        self.annotation_file = os.path.join(self.annotation_folder, 'instances_{}{}.json'.format('train' if train else 'val', year))

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

        if os.path.exists(self.annotation_file) and os.path.getsize(self.annotation_file) > 0:
            with open(self.annotation_file) as f:
                data = json.load(f)
                annotations = data['annotations']
                image_infos = {img['id']: img for img in data['images']}

            self.images = []
            self.annotations = []

            for img_id, img_info in image_infos.items():
                image_path = os.path.join(self.image_folder, img_info['file_name'])
                if os.path.isfile(image_path):
                    self.images.append(image_path)
                    img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
                    self.annotations.append(img_annotations)
        else:
            raise RuntimeError(f"Annotation file not found or is empty: {self.annotation_file}")

    def download(self, resources):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        for key in ['train', 'val', 'annotations']:
            url, md5 = resources[self.year][key]
            download_and_extract_archive(url, download_root=self.root, md5=md5)

    def _check_exists(self):
        # Check if dataset, images and annotations directories exist
        return os.path.exists(self.image_folder) and os.path.exists(self.annotation_folder)

    def __getitem__(self, index):
        # Load the image
        img_info = self.images[index]
        img_id = img_info['id']
        img_path = os.path.join(self.image_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Extract annotations for the given image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Extract bounding boxes and labels
        bboxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']
            label = ann['category_id']
            # Convert bbox coordinates from COCO to [xmin, ymin, xmax, ymax] format
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bboxes.append(bbox)
            labels.append(label)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        target = {'labels': labels, 'boxes': bboxes}
        return img, target

    def __len__(self):
        return len(self.annotations)

# Usage example:
# dataset = COCODataset(root='path/to/coco', year='2017', train=True, download=True)
