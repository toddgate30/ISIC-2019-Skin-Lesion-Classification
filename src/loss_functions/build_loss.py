import torch
import torch.nn as nn
import numpy as np


def build_loss(config, class_counts):
    loss_name = config.get("loss_function", "CrossEntropy")
    if loss_name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "WeightedCrossEntropy":
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        return nn.CrossEntropyLoss(weights=class_weights)
    else:
        raise NotImplementedError(f"Loss Function {loss_name} not implemented.")