import os
import torch
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
from PIL import Image
from urllib.error import URLError

class VOCDataset(VisionDataset):
    def __init__(self, root, year, train=True, download=False, transform=None):
        super(VOCDataset, self).__init__(root, transform=transform)
        self.year = year  # The year of the VOC dataset
        self.train = train  # Whether to load the training set

        # Define the paths for the data, images, annotations, and image sets
        self.data_folder = os.path.join(root, 'VOC{}'.format(year))
        self.image_folder = os.path.join(self.data_folder, 'JPEGImages')
        self.annotation_folder = os.path.join(self.data_folder, 'Annotations')
        self.image_sets_path = os.path.join(self.data_folder, 'ImageSets', 'Main', '{}.txt'.format('trainval' if train else 'test'))

        # Define the resources for downloading the VOC dataset
        resources = {
            '2007': {
                'trainval': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', 'a442f751799d7e0974a3d4f5c8b5b5b3'),
                'test': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', 'b6e924de25625d8de591ea690078ad9f')
            },
            '2012': {
                'trainval': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd'),
                'test': ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOC2012test.tar', 'e14f763270cf193d291bd2c1a9a1236d')
            }
        }

        # Download the dataset if specified
        if download:
            self.download(resources)

        # Initialize lists for storing image and annotation paths
        self.images = []
        self.annotations = []
        with open(image_sets_path) as f:
            image_ids = f.readlines()
        image_ids = [x.strip() for x in image_ids]

        # Add the paths of the images and annotations to the lists
        for img_id in image_ids:
            image_path = os.path.join(self.image_folder, img_id + '.jpg')
            annotation_path = os.path.join(self.annotation_folder, img_id + '.xml')

            if os.path.isfile(image_path) and os.path.isfile(annotation_path):
                self.images.append(image_path)
                self.annotations.append(annotation_path)

    def download(self, resources):
        # Check if the dataset already exists
        if self._check_exists():
            return

        # Create the root directory if it does not exist
        os.makedirs(self.root, exist_ok=True)

        # Define the URL and filename for downloading the dataset
        trainval_or_test = 'trainval' if self.train else 'test'
        url, md5 = resources[self.year][trainval_or_test]
        filename = 'VOC{}'.format(self.year)

        # Download and extract the dataset
        try:
            print(f"Downloading {url}")
            download_and_extract_archive(url, download_root=self.root, filename=filename, md5=self.md5)
        except URLError as error:
            print(f"Failed to download:\n{error}")
            raise RuntimeError(f"Error downloading {filename}")

    def _check_exists(self):
        # Check if dataset, images and annotations directories exist
        return os.path.exists(self.data_folder) and \
               os.path.exists(self.image_folder) and \
               os.path.exists(self.annotation_folder)

    def __getitem__(self, index):
        # Load the image
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')

        # Load the annotations
        annotation_path = self.annotations[index]
        with open(annotation_path) as file:
            xml_tree = ET.parse(file)
        root = xml_tree.getroot()

        # Extract relevant data from the XML file (e.g., bounding boxes)
        target = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            label = obj.find('name').text
            # Convert bbox coordinates from strings to integers
            bbox = [int(bbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            
            # Append label and bounding box to target list
            target.append({'label': label, 'bbox': bbox})

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        # Return the number of samples in dataset
        return len(self.images)

