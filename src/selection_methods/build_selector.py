import torch
from .uniform_selector import uniform_selector
from .div_bs import div_bs

def build_selector(config):
    selection_method = config.get("selection_method", "Uniform")
    if selection_method == "Uniform":
        return uniform_selector
    elif selection_method == "DivBS":
        return div_bs
    else:
        raise NotImplementedError(f"Selection Method {selection_method} has not been implemented")