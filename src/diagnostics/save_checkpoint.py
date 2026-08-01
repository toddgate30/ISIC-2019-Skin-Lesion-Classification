import torch
from pathlib import Path
import numpy as np
import random

def save_checkpoint(config, context, epoch, step, best_val_accuracy):
    """
    Save the model checkpoint.

    Parameters
    ----------
    context : TrainingContext
        The training context containing the model and other training components.
    
    checkpoint_dir : str
        Directory where the checkpoint will be saved.
    """

    # Ensure the checkpoint directory exists
    checkpoint_dir = Path(config["run_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Define the checkpoint file path
    temp_path = checkpoint_dir / "temp_checkpoint.pth"
    checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"

    checkpoint = {
        "config": config,

        "model_state_dict": context.model.state_dict(),
        "optimizer_state_dict": context.optimizer.state_dict(),
        "loss_function_state_dict": context.loss_function.state_dict(),

        "scheduler": (context.lr_scheduler.state_dict() if context.lr_scheduler is not None else None),

        "epoch": epoch,
        "step": step,
        "best_val_accuracy": best_val_accuracy,

        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    # Save the checkpoint
    torch.save(checkpoint, temp_path)
    temp_path.rename(checkpoint_path)
