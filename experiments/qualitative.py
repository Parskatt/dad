from pathlib import Path
from argparse import ArgumentParser

from dad.utils import visualize_keypoints
from dad.detectors import all_detectors, load_detector_by_name

if __name__ == "__main__":
    parser = ArgumentParser("Qualitative comparisons.")
    parser.add_argument(
        "--detector",
        choices=all_detectors,
        required = True,
    )
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--weights_path", required=False)
    parser.add_argument("--num_keypoints", required=False, default=2048, type=int)

    args = parser.parse_args()
    vis_path = Path("vis") / args.detector / str(args.num_keypoints) / args.image_path

    detector = load_detector_by_name(args.detector, weights_path=args.weights_path)
    visualize_keypoints(args.image_path, vis_path, detector, args.num_keypoints)