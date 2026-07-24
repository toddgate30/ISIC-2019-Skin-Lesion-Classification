import torch
from math import floor

def build_selector(config):
    selection_method = config.get("selection_method", "Uniform")
    if selection_method == "Uniform":
        return uniform_selector
    else:
        raise NotImplementedError(f"Selection Method {selection_method} has not been implemented")

def uniform_selector(metabatch_images, metabatch_labels, ratio):
    metabatch_size = len(metabatch_labels)
    batch_size = floor(metabatch_size * ratio)

    selected_indices = torch.randperm(metabatch_size)[:batch_size]

    return metabatch_images[selected_indices], metabatch_labels[selected_indices]