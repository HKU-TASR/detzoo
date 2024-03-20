from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    """
    Custom collate function for object detection tasks.

    This function separates images and targets from the batch, uses the default collate
    function to collate the images, and leaves the targets uncollated because each target
    can have a different number of bounding boxes.

    Args:
        batch (list): A list of samples, where each sample is a tuple (image, target).

    Returns:
        A tuple (images, targets), where:
            - images is a tensor obtained by collating the images from the batch.
            - targets is a list of targets from the batch.
    """

    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Use the default collate function to collate the images
    images = default_collate(images)

    # The targets do not need any special treatment, so they are returned as is.
    # This is because each target can have a different number of bounding boxes,
    # so they cannot be stacked into a tensor.
    return images, targets