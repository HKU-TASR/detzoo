import torch
from torchvision import models

def load_pretrained_model(model_name, path, num_classes, num_layers_to_freeze):
    """
    Load a pretrained model from a .pt file and freeze the first num_layers_to_freeze layers.

    Args:
    model_name: The name of the model (e.g., 'vgg16', 'resnet50').
    path: The path to the .pt file.
    num_classes: The number of output classes.
    num_layers_to_freeze: The number of layers to freeze.

    Returns:
    model: The modified pretrained model.
    """
    # Initialize the appropriate model
    if model_name.lower() == 'vgg16':
        model = models.vgg16()
    elif model_name.lower() == 'resnet50':
        model = models.resnet50()
    else:
        raise ValueError("Invalid model name. Expected 'vgg16' or 'resnet50'.")

    # Load the state dict from the .pt file
    state_dict = torch.load(path)

    # Update the model's state dict to use the loaded state dict
    model.load_state_dict(state_dict)

    # Freeze the first num_layers_to_freeze layers
    for i, param in enumerate(model.parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False

    # Update the final layer to have the correct number of classes
    if model_name.lower() == 'vgg16':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name.lower() == 'resnet50':
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model