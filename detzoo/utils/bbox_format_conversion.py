import torch

def bbox_to_yolo_format(bbox, image_size, S=7, B=2, C=20):
    """
    Convert bbox format target to YOLO format.

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

def yolo_to_bbox_format(yolo, image_size, S=7, B=2, C=20):
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