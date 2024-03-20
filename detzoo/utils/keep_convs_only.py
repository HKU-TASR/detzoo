from torch import nn

def keep_convs_only(backbone):

    '''
    Remove all layers after the last conv layer in the backbone.

    Args:
    backbone: The backbone model

    Returns:
    backbone: The backbone model with only conv layers
    '''

    # Flatten all children modules
    children = list(backbone.children())
    i = 0
    while i < len(children):
        if isinstance(children[i], nn.Sequential):
            children[i:i+1] = list(children[i].children())
        else:
            i += 1

    # Find the last conv layer
    last_conv_idx = None
    for idx, module in enumerate(children):
        if isinstance(module, nn.Conv2d):
            last_conv_idx = idx

    if last_conv_idx is None:
        raise ValueError("No Conv2d layer found in backbone")

    # Remove layers after the last conv layer
    children = children[:last_conv_idx+1]

    out_channels = children[-1].out_channels
    backbone = nn.Sequential(*children)

    return backbone, out_channels