import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def build_model(config):
    model_name = config.get("model", "resnet18")
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 8)
        return model
    else:
        raise NotImplementedError(f"model {model_name} has not been implemented")
    