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
         "data_params.transform_rotation": [0, 10, 20, 30, 40, 50]
    }
    project_dir = Path("./experiments/supercomputer_tests")
    time = "24:00:00"

    parameter_names = list(grid_search.keys())
    parameter_values = list(grid_search.values())

    for run_id, values in enumerate(itertools.product(*parameter_values)):
        # Make an independent copy for this run
        config = copy.deepcopy(base_config)

        # Apply this combination of hyperparameters
        for parameter_name, value in zip(parameter_names, values):
            set_nested_value(config, parameter_name, value)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_name = f"{timestamp}_rotate={config['data_params']['transform_rotation']}"
        submit_run(config, project_dir, job_name, time)

if __name__ == "__main__":
    run()