import torch
import torch.nn as nn


def build_loss(config):
    loss_name = config.get("loss_function", "CrossEntropy")
    if loss_name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss Function {loss_name} not implemented.")