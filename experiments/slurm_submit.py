import argparse
import subprocess
from dad.utils import wrap_in_sbatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment using SLURM")
    parser.add_argument(
        "experiment_path", help="Experiment filepath"
    )
    parser.add_argument(
        "--account", "-a", required=True, help="SLURM account number to use"
    )
    parser.add_argument(
        "--time",
        "-t",
        default="2-23:00:00",
        help="Time allocation in SLURM format (default: 2-23:00:00)",
    )
    args = parser.parse_args()

    sbatch_command = wrap_in_sbatch(
        f"python {args.experiment_path}",
        account = args.account,
        time_alloc = args.time,
    )
    subprocess.run(["sbatch"], input=sbatch_command, text=True)
