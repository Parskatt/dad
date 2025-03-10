import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import dad
import wandb
from dad.benchmarks import (
    HPatchesIllum,
    HPatchesViewpoint,
    Mega1500,
    MegaIMCPT,
    NumInliersBenchmark,
    ScanNet1500,
)
from dad.loss import MaxDistillLoss
from dad.train import train_k_steps
from dad.types import Benchmark
from dad.checkpoint import CheckPoint
from dad.datasets.megadepth import MegadepthBuilder
from dad.matchers import load_roma_matcher
from dad.utils import (
    get_data_iterator,
    setup_experiment,
    run_benchmarks,
    run_qualitative_examples,
)


def main():
    root_workspace_path = "workspace"
    train_image_size = 640
    train_batch_size = 8
    weight_decay = 1e-5
    lr = 2e-4
    n0, N, eval_every_nth = 0, 100_000, 200
    test_num_keypoints = 512
    num_eval_batches = eval_every_nth
    rot_360 = True
    shake_t = 32
    workspace_path = setup_experiment(
        __file__, root_workspace_path=root_workspace_path, disable_wandb=dad.DEBUG_MODE
    )

    test_num_keypoints = 512

    student = dad.load_DaD(pretrained=False)

    params = [
        {"params": student.parameters(), "lr": lr},
    ]
    optim = AdamW(params, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optim, T_max=N)
    checkpointer = CheckPoint(workspace_path)

    mega = MegadepthBuilder(data_root="data/megadepth")

    grad_scaler = torch.amp.GradScaler()

    roma_matcher = load_roma_matcher()

    mega_test_data = mega.dedode_test_split(
        shake_t=shake_t,
        image_size=train_image_size,
    )

    mega_test = NumInliersBenchmark(
        mega_test_data,
        num_samples=num_eval_batches,
        num_keypoints=test_num_keypoints,
    )

    megadepth_train = mega.dedode_train_split(
        rot_360=rot_360,
        shake_t=shake_t,
        image_size=train_image_size,
    )
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)

    benchmarks: list[Benchmark] = [
        Mega1500,
        MegaIMCPT,
        HPatchesViewpoint,
        HPatchesIllum,
        ScanNet1500,
    ]
    run_qualitative_examples(
        model=student,
        workspace_path=workspace_path,
        test_num_keypoints=test_num_keypoints,
    )
    teacher_light = dad.load_DaDLight()
    teacher_dark = dad.load_DaDDark()
    loss = MaxDistillLoss(teacher_light, teacher_dark)
    for n in range(n0, N, eval_every_nth):
        mega_dataloader = get_data_iterator(
            megadepth_train, mega_ws, train_batch_size, eval_every_nth
        )
        train_k_steps(
            n,
            eval_every_nth,
            mega_dataloader,
            student,
            loss,
            optim,
            lr_scheduler,
            grad_scaler=grad_scaler,
        )
        checkpointer.save(student, optim, lr_scheduler, n)
        wandb.log(
            mega_test.benchmark(student),
            step=dad.GLOBAL_STEP,
        )
        run_qualitative_examples(
            model=student,
            workspace_path=workspace_path,
            test_num_keypoints=test_num_keypoints,
        )
    run_benchmarks(
        benchmarks, roma_matcher, student, step=dad.GLOBAL_STEP + 1, sample_every=1
    )


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "16"
    main()
