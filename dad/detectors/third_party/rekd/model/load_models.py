import torch
from .REKD import REKD


def load_detector(args, device):
    args.group_size, args.dim_first, args.dim_second, args.dim_third = model_parsing(
        args
    )
    model1 = REKD(args, device)
    model1.load_state_dict(torch.load(args.load_dir, weights_only=True))
    model1.export()
    model1.eval()
    model1.to(device)  ## use GPU

    return model1


## Load our model
def model_parsing(args):
    group_size = args.load_dir.split("_group")[1].split("_")[0]
    dim_first = args.load_dir.split("_f")[1].split("_")[0]
    dim_second = args.load_dir.split("_s")[1].split("_")[0]
    dim_third = args.load_dir.split("_t")[1].split(".log")[0]

    return int(group_size), int(dim_first), int(dim_second), int(dim_third)
