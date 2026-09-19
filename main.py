import torch
import wandb
import argparse
import yaml
from datetime import datetime
import warnings
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Callable
import random
import numpy as np
from pathlib import Path

from src.data.prepare_data import prepare_data
from src.models.build_model import build_model
from src.selection_methods.build_selector import build_selector
from src.Trainer import Trainer
from src.loss_functions.build_loss import build_loss
from src.optimizers.build_optimizer import build_optimizer
from src.diagnostics.diagnostic_manager import DiagnosticManager
from src.schedulers.build_scheduler import build_scheduler

@dataclass
class TrainingContext:
    """Stores the components and data loaders required for model training.

    Attributes:
        model: Model being trained.
        optimizer: Optimizer used to update the model parameters.
        loss_function: Loss function used during training.
        device: Device on which the model and training computations are performed.
        train_loader: Data loader used to provide training batches.
        train_metrics_loader: Data loader used to compute training metrics.
        val_loader: Data loader used for validation.
        selector: Batch-selection method used to select training examples.
        lr_scheduler: Learning-rate scheduler associated with the optimizer.
        class_names: Names of the classes in the dataset.
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    device: torch.device
    train_loader: DataLoader
    train_metrics_loader: DataLoader
    val_loader: DataLoader
    selector: Callable
    lr_scheduler: torch.optim.lr_scheduler
    class_names: list

def parse_args():
    """Parse command-line arguments used to configure a training run.

    Returns:
        Parsed command-line arguments containing the configuration file path
        and run directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration yaml file"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory where the run and main.py are located"
    )
    return parser.parse_args()

def restore_checkpoint(context, checkpoint_path, trainer):
    """Restore training state from a saved checkpoint.

    Restores the model, optimizer, loss function, learning-rate scheduler,
    random number generator states, and trainer state. The checkpoint's
    configuration is returned so that the training run can resume using the
    configuration with which the checkpoint was created.

    Args:
        context: Training context containing the model, optimizer, loss
            function, and learning-rate scheduler to restore.
        checkpoint_path: Path to the checkpoint file.
        trainer: Trainer whose training state should be restored.

    Returns:
        The configuration saved in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=context.device)

    config = checkpoint["config"]

    context.model.load_state_dict(checkpoint["model_state_dict"])
    context.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    context.loss_function.load_state_dict(checkpoint["loss_function_state_dict"])

    if context.lr_scheduler is not None and checkpoint["scheduler"] is not None:
        context.lr_scheduler.load_state_dict(checkpoint["scheduler"])

    # Restore RNG states
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    trainer.load_state(checkpoint["epoch"], checkpoint["step"], checkpoint["best_val_accuracy"])
    return config


def main():
    """Initialize and run a training experiment.

    Loads the experiment configuration, prepares the data and training
    components, restores a previous checkpoint when available, and runs the
    training process to completion. New runs are initialized in Weights &
    Biases, while resumed runs continue the existing Weights & Biases run.
    """
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["run_dir"] = args.run_dir

    

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        warnings.warn("No cuda device found. Training on CPU")
        device = torch.device("cpu")

    train_loader, train_metrics_loader, val_loader, class_counts, class_names = prepare_data(config)
    model = build_model(config).to(device)
    loss_function = build_loss(config, class_counts).to(device)
    selector = build_selector(config)
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)

    context = TrainingContext(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        train_loader=train_loader,
        train_metrics_loader=train_metrics_loader,
        val_loader=val_loader,
        selector=selector,
        lr_scheduler=lr_scheduler,
        class_names=class_names
    )

    diagnostic_manager = DiagnosticManager(config)
    trainer = Trainer(context, diagnostic_manager, config)

    checkpoint_path = Path(config["run_dir"]) / "checkpoints" / "latest_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"Restoring from checkpoint: {checkpoint_path}")
        config = restore_checkpoint(context, checkpoint_path, trainer)
        wandb_name = config["wandb"].get("name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run = wandb.init(
            project=config["wandb"]["project"],
            entity="toddgate30-byu",
            config=config,
            name=wandb_name,
            id=config['wandb'].get('run_id', None),
            resume="allow"
        )
    else:
        wandb_name = config["wandb"].get("name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run = wandb.init(
                project=config["wandb"]["project"],
                entity="toddgate30-byu",
                config=config,
                name=wandb_name
            )
        config['wandb']['run_id'] = run.id
        trainer.before_train()
    
    trainer.train()
    trainer.after_train()

    wandb.finish()

if __name__ == "__main__":
    main()