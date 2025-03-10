import argparse

## for fix seed
import random
import torch
import numpy


def get_config(jupyter=False):
    parser = argparse.ArgumentParser(description="Train REKD Architecture")

    ## basic configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../ImageNet2012/ILSVRC2012_img_val",  # default='path-to-ImageNet',
        help="The root path to the data from which the synthetic dataset will be created.",
    )
    parser.add_argument(
        "--synth_dir",
        type=str,
        default="",
        help="The path to save the generated sythetic image pairs.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="trained_models/weights",
        help="The path to save the REKD weights.",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="Set saved model parameters if resume training is desired.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="REKD",
        help="The Rotaton-equivaraiant Keypoint Detection (REKD) experiment name",
    )
    ## network architecture
    parser.add_argument(
        "--factor_scaling_pyramid",
        type=float,
        default=1.2,
        help="The scale factor between the multi-scale pyramid levels in the architecture.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=36,
        help="The number of groups for the group convolution.",
    )
    parser.add_argument(
        "--dim_first",
        type=int,
        default=2,
        help="The number of channels of the first layer",
    )
    parser.add_argument(
        "--dim_second",
        type=int,
        default=2,
        help="The number of channels of the second layer",
    )
    parser.add_argument(
        "--dim_third",
        type=int,
        default=2,
        help="The number of channels of the thrid layer",
    )
    ## network training
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs for training."
    )
    ## Loss function
    parser.add_argument(
        "--init_initial_learning_rate",
        type=float,
        default=1e-3,
        help="The init initial learning rate value.",
    )
    parser.add_argument(
        "--MSIP_sizes", type=str, default="8,16,24,32,40", help="MSIP sizes."
    )
    parser.add_argument(
        "--MSIP_factor_loss",
        type=str,
        default="256.0,64.0,16.0,4.0,1.0",
        help="MSIP loss balancing parameters.",
    )
    parser.add_argument("--ori_loss_balance", type=float, default=100.0, help="")
    ## Dataset generation
    parser.add_argument(
        "--patch_size",
        type=int,
        default=192,
        help="The patch size of the generated dataset.",
    )
    parser.add_argument(
        "--max_angle",
        type=int,
        default=180,
        help="The max angle value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--min_scale",
        type=float,
        default=1.0,
        help="The min scale value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=1.0,
        help="The max scale value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--max_shearing",
        type=float,
        default=0.0,
        help="The max shearing value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--num_training_data",
        type=int,
        default=9000,
        help="The number of the generated dataset.",
    )
    parser.add_argument(
        "--is_debugging",
        type=bool,
        default=False,
        help="Set variable to True if you desire to train network on a smaller dataset.",
    )
    ## For eval/inference
    parser.add_argument(
        "--num_points",
        type=int,
        default=1500,
        help="the number of points at evaluation time.",
    )
    parser.add_argument(
        "--pyramid_levels", type=int, default=5, help="downsampling pyramid levels."
    )
    parser.add_argument(
        "--upsampled_levels", type=int, default=2, help="upsampling image levels."
    )
    parser.add_argument(
        "--nms_size",
        type=int,
        default=15,
        help="The NMS size for computing the validation repeatability.",
    )
    parser.add_argument(
        "--border_size",
        type=int,
        default=15,
        help="The number of pixels to remove from the borders to compute the repeatability.",
    )
    ## For HPatches evaluation
    parser.add_argument(
        "--hpatches_path",
        type=str,
        default="./datasets/hpatches-sequences-release",
        help="dataset ",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="debug",
        help="debug, view, illum, full, debug_view, debug_illum ...",
    )
    parser.add_argument(
        "--descriptor", type=str, default="hardnet", help="hardnet, sosnet, hynet"
    )

    args, weird_args = (
        parser.parse_known_args() if not jupyter else parser.parse_args(args=[])
    )

    fix_randseed(12345)

    if args.synth_dir == "":
        args.synth_dir = "datasets/synth_data"

    args.MSIP_sizes = [int(i) for i in args.MSIP_sizes.split(",")]
    args.MSIP_factor_loss = [float(i) for i in args.MSIP_factor_loss.split(",")]

    return args


def fix_randseed(randseed):
    r"""Fix random seed"""
    random.seed(randseed)
    numpy.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    # torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
