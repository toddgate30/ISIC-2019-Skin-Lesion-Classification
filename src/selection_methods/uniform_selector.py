import torch
from math import floor

def uniform_selector(metabatch_images, metabatch_labels, ratio, context):
    """Select a random subset of a meta-batch using uniform sampling.

    Args:
        metabatch_images: Images contained in the meta-batch.
        metabatch_labels: Labels corresponding to the meta-batch images.
        ratio: Fraction of the meta-batch to select.
        context: Not used in this function

    Returns:
        A tuple containing the selected images and their corresponding labels.
    """
    metabatch_size = len(metabatch_labels)
    batch_size = floor(metabatch_size * ratio)

    selected_indices = torch.randperm(metabatch_size)[:batch_size]

    return metabatch_images[selected_indices], metabatch_labels[selected_indices]