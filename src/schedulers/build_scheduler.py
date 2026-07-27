import torch

def build_scheduler(optimizer, config):
    scheduler_name = config.get("scheduler", None)
    kwargs = config.get("scheduler_params", {})

    if scheduler_name is None:
        pass
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("total_epochs", 100), **kwargs)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} has not been implemented")