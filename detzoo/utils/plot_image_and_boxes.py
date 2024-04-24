import torch
import os
import utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_and_boxes(image, target, classes, yolo_format=False, S=7, B=2):
    # image: [H, W, C]
    # target: {'boxes': (N, 4), 'confidences': (N, ), 'labels': (N,)}
    
    width, height = image.shape[1], image.shape[0]
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis('off')

    # Define a list of colors for different classes
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    category_color = {}
    color_index = 0

    # Get the bounding boxes and labels from the target
    boxes = target['boxes']
    confidences = target['confidences']
    labels = target['labels']

    # Plot each bounding box
    for box, confidence, label in zip(boxes, confidences, labels):
        xmin, ymin, xmax, ymax = box
        if label.item() in category_color.keys():
            color = category_color[label.item()]
        else:
            color = colors[color_index]
            category_color[label.item()] = color
            color_index = (color_index + 1) % len(colors)
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        name = classes[label.item()]
        plt.text(xmin, ymin, f'{name} {confidence:.2f}', color=color)

    # Show the figure
    plt.show()



