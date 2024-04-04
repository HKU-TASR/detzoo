import pytest
from torchvision import transforms
from datasets.voc import VOCDataset
from datasets.coco import COCODataset

@pytest.fixture(params=[('VOC', '2007', True), ('VOC', '2007', False), ('VOC', '2012', True), 
                        ('COCO', '2017', True), ('COCO', '2017', False)])
def dataset(request):
    dataset_type, year, train = request.param
    root = '/home/lujialin/data'
    if dataset_type == 'VOC':
        return VOCDataset(root=root, year=year, train=train, transform=transforms.ToTensor())
    elif dataset_type == 'COCO':
        return COCODataset(root=root, year=year, train=train, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")


def test_dataset_loading(request, dataset):
    assert len(dataset) > 0, "Dataset is empty"

def test_dataset_images(request, dataset):
    for i in range(min(5, len(dataset))):
        if isinstance(dataset, VOCDataset):
            assert dataset.images[i] is not None, f"Image path at index {i} is None in VOC Dataset"
        elif isinstance(dataset, COCODataset):
            img_info = dataset.images[i]
            assert img_info is not None, f"Image info at index {i} is None in COCO Dataset"
            img_path = os.path.join(dataset.image_folder, img_info['file_name'])
            assert os.path.exists(img_path), f"Image file at index {i} does not exist in COCO Dataset"

def test_dataset_annotations(request, dataset):
    for i in range(min(5, len(dataset))):
        if isinstance(dataset, VOCDataset):
            assert dataset.annotations[i] is not None, f"Annotation at index {i} is None in VOC Dataset"
        elif isinstance(dataset, COCODataset):
            ann = dataset.annotations[i]
            assert ann is not None, f"Annotation at index {i} is None in COCO Dataset"

def test_dataset_items(request, dataset):
    for i in range(min(10, len(dataset))):
        img, target = dataset[i]
        assert img is not None, f"Image at index {i} is None"
        assert target is not None, f"Target at index {i} is None"
