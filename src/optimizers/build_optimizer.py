import torch

def build_optimizer(model, config):
    optim_name = config.get("optimizer", "AdamW")
    kwargs = config.get("optim_params", {})

    if optim_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optim_name} has not been implemented")