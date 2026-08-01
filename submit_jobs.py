from pathlib import Path
import subprocess
import yaml
import shlex


def submit_run(
    config: dict,
    project_dir: Path,
    job_name: str,
    time: str = "24:00:00",
):
    """
    Save a config dictionary and submit a SLURM job.

    Parameters
    ----------
    batch_id : str
        Unique identifier for this experiment batch.

    config : dict
        Configuration dictionary to save as YAML.

    project_dir : Path
        Directory containing the project and main.py.

    log_dir : Path
        Directory where SLURM stdout/stderr logs will be saved.

    wandb_dir : Path
        Directory where offline W&B runs will be saved.

    job_name : str
        Name for the SLURM job.

    time : str
        SLURM time limit.
    """

    # Make sure directories exist
    run_dir = project_dir / job_name

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    wandb_dir = run_dir / "wandb"
    log_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = run_dir / "config.yaml"

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Create SLURM script
    slurm_flags = [
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=16G",
        f"--time={time}",
        f"--job-name={job_name}",
        f"--output={log_dir}/%j.out",
        f"--error={log_dir}/%j.err",
        "--qos=standby",
        "--requeue",
        "--exclude=cs-1-[1-5],cs-2-[1-2],cs-3-1",
    ]
    # "--requeue",

    python_cmd = [
        "env",
        "WANDB_MODE=offline",
        f"WANDB_DIR={wandb_dir}",
        "/home/todd30ap/.conda/envs/online-bs-preemptible/bin/python",
        "main.py",
        "--config",
        str(config_path),
        "--run_dir",
        str(run_dir)
    ]

    # result = subprocess.run(["srun"] + slurm_flags + python_cmd, check=True)

    result = subprocess.run(
        ["sbatch"] + slurm_flags + [f"--wrap={shlex.join(python_cmd)}"],
        check=True,
    )

    # Submit job
    # result = subprocess.run(
    #     ["sbatch", str(slurm_path)],
    #     capture_output=True,
    #     text=True,
    #     check=True,
    # )

    # print(f"Submitted run: {result.stdout.strip()}")

def set_nested_value(config, parameter_path, value):
    keys = parameter_path.split(".")

    current = config

    for key in keys[:-1]:
        if key not in current:
            raise KeyError(
                f"Could not find '{key}' while setting '{parameter_path}'"
            )

        current = current[key]

    final_key = keys[-1]

    if final_key not in current:
        raise KeyError(
            f"Could not find '{parameter_path}' in config"
        )

    current[final_key] = value