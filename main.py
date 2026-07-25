import torch
import wandb
import argparse
import yaml

from src.data.prepare_data import prepare_data
from src.models.build_model import build_model
from src.selection_methods.build_selector import build_selector
from src.Trainer import Trainer
from src.loss_functions.build_loss import build_loss
from src.optimizers.build_optimizer import build_optimizer
from src.diagnostics.diagnostic_manager import DiagnosticManager
from datetime import datetime
import warnings
from dataclasses import dataclass
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration yaml file"
    )
    return parser.parse_args()

@dataclass
class TrainingContext:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    device: torch.device
    train_loader: DataLoader
    train_metrics_loader: DataLoader
    val_loader: DataLoader

def main():

    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "name" in config["wandb"].keys():
        wandb_name = config["wandb"]["name"]
        config["run_dir"] = f"{timestamp}_{wandb_name}"
    else:
        wandb_name = timestamp
        config["run_dir"] = wandb_name

    wandb.init(
        project=config["wandb"]["project"],
        entity="toddgate30-byu",
        config=config,
        name=wandb_name
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        warnings.warn("No cuda device found. Training on CPU")
        device = torch.device("cpu")



    train_loader, train_metrics_loader, val_loader, class_counts = prepare_data(config)

    model = build_model(config)

    selector = build_selector(config)

    loss_function = build_loss(config, class_counts, device)

    optimizer = build_optimizer(model, config)

    diagnostic_manager = DiagnosticManager(config)

    trainer = Trainer(
        model=model,
        dataloaders=(train_loader, train_metrics_loader, val_loader),
        selector=selector,
        loss_function=loss_function,
        optimizer = optimizer,
        diagnostic_manager = diagnostic_manager,
        config=config,
        device=device
        )

    trainer.before_train()
    trainer.train()
    trainer.after_train()

    wandb.finish()

if __name__ == "__main__":
    main()