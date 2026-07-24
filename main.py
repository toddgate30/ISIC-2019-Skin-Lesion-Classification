import torch
import wandb
import argparse
import yaml

from src.data.prepare_data import prepare_data
from src.models.build_model import build_model
from src.selection_methods.build_selector import build_selector
from src.Trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        trype=str,
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
        project=config["wandb"]["project"]
        config=config
    )

    dataloaders = prepare_data(config)

    model = build_model(config)

    selector = build_selector(config)

    trainer = Trainer(
        model=model,
        selector=selector,
        dataloaders=dataloaders,
        config=config)
    
    trainer.before_train()
    trainer.train()
    trainer.after_train()

if __name__ == "__main__":
    main()