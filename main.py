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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration yaml file"
    )
    return parser.parse_args()

def main():

    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # if torch.cuda.is_available():
    #     config["device"] = torch.device("cuda")
    # else:


    wandb.init(
        project=config["wandb"]["project"],
        entity="toddgate30-byu",
        config=config
    )

    dataloaders = prepare_data(config)

    model = build_model(config)

    selector = build_selector(config)

    loss_function = build_loss(config)

    optimizer = build_optimizer(model, config)

    diagnostic_manager = DiagnosticManager(config)

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        selector=selector,
        loss_function=loss_function,
        optimizer = optimizer,
        diagnostic_manager = diagnostic_manager,
        config=config
        )

    trainer.before_train()
    trainer.train()
    trainer.after_train()

    wandb.finish()

if __name__ == "__main__":
    main()