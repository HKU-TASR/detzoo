import torch

def iou(box1, box2, format='ltrb'):
    """
    Compute the Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
    box1, box2: Tensors of shape (N, 4) representing bounding boxes.
    format: A string indicating the format of the coordinates ('ltrb', 'ltwh', 'ccwh').
    - ltrb: (left, top, right, bottom)
    - ltwh: (left, top, width, height)
    - ccwh: (center_x, center_y, width, height)

    Returns:
    ious: Tensor of shape (N,) representing the IoU for each bounding box
    """
    if format == 'ltrb':
        pass
    elif format == 'ltwh':
        box1 = torch.stack([box1[:, 0], box1[:, 1], box1[:, 0] + box1[:, 2], box1[:, 1] + box1[:, 3]], dim=1)
        box2 = torch.stack([box2[:, 0], box2[:, 1], box2[:, 0] + box2[:, 2], box2[:, 1] + box2[:, 3]], dim=1)
    elif format == 'ccwh':
        box1 = torch.stack([box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2, box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2], dim=1)
        box2 = torch.stack([box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2, box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2], dim=1)
    else:
        raise ValueError("Invalid format. Expected 'ltrb', 'ltwh', or 'ccwh'.")

    # Compute the coordinates of the intersection rectangle
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    # Compute the area of intersection rectangle
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Compute the area of each bounding box
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Compute the area of union
    union = box1_area + box2_area - intersection

    # Compute IoU
    ious = intersection / union

    return ious