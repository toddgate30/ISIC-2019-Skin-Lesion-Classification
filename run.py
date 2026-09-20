import yaml
import itertools
import copy
from datetime import datetime
from pathlib import Path
from submit_jobs import submit_run, set_nested_value

def run():
    config_dir = "config.yaml"

    with open(config_dir, "r") as file:
            base_config = yaml.safe_load(file)

    grid_search = {
         "selection_method": ['Uniform', 'DivBS'],
         "loss_function": ['CrossEntropy', 'WeightedCrossEntropy']
    }
    project_dir = Path("./experiments/Online-Batch_Selection/9-19-26")
    time = "24:00:00"

    parameter_names = list(grid_search.keys())
    parameter_values = list(grid_search.values())

    for run_id, values in enumerate(itertools.product(*parameter_values)):
        # Make an independent copy for this run
        config = copy.deepcopy(base_config)

        # Apply this combination of hyperparameters
        for parameter_name, value in zip(parameter_names, values):
            set_nested_value(config, parameter_name, value)

        config['wandb']['name'] = "_".join([config['selection_method'], config['loss_function'], config['wandb']['name']])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_name = f"{timestamp}_{config['wandb']['name']}"
        submit_run(config, project_dir, job_name, time)

if __name__ == "__main__":
    run()