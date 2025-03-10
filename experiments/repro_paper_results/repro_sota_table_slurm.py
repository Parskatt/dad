import argparse
import subprocess
from dad.utils import wrap_in_sbatch
from dad.benchmarks import all_benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks using SLURM")
    parser.add_argument(
        "--account", "-a", required=True, help="SLURM account number to use"
    )
    parser.add_argument(
        "--time",
        "-t",
        default="2-23:00:00",
        help="Time allocation in SLURM format (default: 2-23:00:00)",
    )
    parser.add_argument('--detectors', '-d', nargs='+', default=["DaD"], help='List of detectors to use')
    args = parser.parse_args()

    for detector in args.detectors:
        for benchmark in all_benchmarks:
            sbatch_command = wrap_in_sbatch(
                f"python experiments/benchmark.py --detector {detector} --benchmark {benchmark}",
                account = args.account,
                time_alloc = args.time,
            )
            subprocess.run(["sbatch"], input=sbatch_command, text=True)
