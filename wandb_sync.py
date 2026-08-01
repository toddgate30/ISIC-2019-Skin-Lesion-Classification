from pathlib import Path
import subprocess
import argparse
import time


def sync_wandb_runs(root_dir):
    root_dir = Path(root_dir).resolve()

    if not root_dir.is_dir():
        raise ValueError(f"Directory does not exist: {root_dir}")

    wandb_dirs = root_dir.rglob("wandb")
    total_runs = 0

    for wandb_dir in wandb_dirs:
        if not wandb_dir.is_dir():
            continue

        runs = [run for run in wandb_dir.iterdir()
                if run.is_dir() and run.name.startswith("offline-run-")]

        for run in runs:
            print(f"Syncing {run}")
            result = subprocess.run(["wandb", "sync", str(run)])

            if result.returncode != 0:
                print(f"Failed to sync {run}")

            total_runs += 1

    print(f"Finished. Attempted to sync {total_runs} runs.")


def main():
    parser = argparse.ArgumentParser(description="Recursively sync W&B offline runs.")
    parser.add_argument("directory", type=Path, help="Root directory containing W&B offline runs.")
    args = parser.parse_args()

    while True:
        sync_wandb_runs(args.directory)
        time.sleep(15)

if __name__ == "__main__":
    main()