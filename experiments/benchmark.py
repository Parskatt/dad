from pathlib import Path
import wandb
import dad
from argparse import ArgumentParser
from dad.benchmarks import all_benchmarks
from dad.matchers import load_roma_matcher
from dad.detectors import all_detectors, load_detector_by_name


if __name__ == "__main__":
    parser = ArgumentParser("SotA comparison benchmarks.")
    parser.add_argument(
        "--detector",
        choices=all_detectors,
    )
    parser.add_argument("--benchmark", choices=all_benchmarks)
    parser.add_argument("--sample_every_kth", required=False, default=1, type=int)
    parser.add_argument("--disable_wandb", required=False, action="store_true")
    parser.add_argument("--num_keypoints", required=False, type=int)
    parser.add_argument("--weights_path", required=False, type=str)

    args = parser.parse_args()
    experiment_name = (
        Path(__file__)
        .relative_to(Path("experiments").absolute())
        .with_suffix("")
        .as_posix()
    )
    roma_matcher = load_roma_matcher()
    wandb.init(
        project="DaD-" + args.benchmark,
        mode="disabled" if args.disable_wandb else "online",
        name=args.detector,
    )
    detector = load_detector_by_name(args.detector, weights_path=args.weights_path)

    bench: dad.Benchmark = getattr(dad.benchmarks, args.benchmark)(
        sample_every=args.sample_every_kth, num_keypoints=args.num_keypoints
    )
    results = bench.benchmark(
        matcher=roma_matcher,
        detector=detector,
    )
    dad.logger.info(results)
    wandb.log(results)
