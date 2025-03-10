import torch
import torch.nn.functional as F
from tqdm import tqdm

from dad.types import Detector
from dad.utils import get_gt_warp, to_best_device


class NumInliersBenchmark:
    def __init__(
        self,
        dataset,
        num_samples=1000,
        batch_size=8,
        num_keypoints=512,
        **kwargs,
    ) -> None:
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.ones(len(dataset)), replacement=False, num_samples=num_samples
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=batch_size, sampler=sampler
        )
        self.dataloader = dataloader
        self.tracked_metrics = {}
        self.batch_size = batch_size
        self.N = len(dataloader)
        self.num_keypoints = num_keypoints

    def compute_batch_metrics(self, outputs, batch):
        kpts_A, kpts_B = outputs["keypoints_A"], outputs["keypoints_B"]
        B, K, H, W = batch["im_A"].shape
        gt_warp_A_to_B, valid_mask_A_to_B = get_gt_warp(
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=H,
            W=W,
        )
        kpts_A_to_B = F.grid_sample(
            gt_warp_A_to_B[..., 2:].float().permute(0, 3, 1, 2),
            kpts_A[..., None, :],
            align_corners=False,
            mode="bilinear",
        )[..., 0].mT
        legit_A_to_B = F.grid_sample(
            valid_mask_A_to_B.reshape(B, 1, H, W),
            kpts_A[..., None, :],
            align_corners=False,
            mode="bilinear",
        )[..., 0, :, 0]
        dists = (
            torch.cdist(kpts_A_to_B, kpts_B).min(dim=-1).values[legit_A_to_B > 0.0]
        ).float()
        if legit_A_to_B.sum() == 0:
            return
        percent_inliers_at_1 = (dists < 0.02).float().mean()
        percent_inliers_at_05 = (dists < 0.01).float().mean()
        percent_inliers_at_025 = (dists < 0.005).float().mean()
        percent_inliers_at_01 = (dists < 0.002).float().mean()
        percent_inliers_at_005 = (dists < 0.001).float().mean()

        self.tracked_metrics["percent_inliers_at_1"] = (
            self.tracked_metrics.get("percent_inliers_at_1", 0)
            + 1 / self.N * percent_inliers_at_1
        )
        self.tracked_metrics["percent_inliers_at_05"] = (
            self.tracked_metrics.get("percent_inliers_at_05", 0)
            + 1 / self.N * percent_inliers_at_05
        )
        self.tracked_metrics["percent_inliers_at_025"] = (
            self.tracked_metrics.get("percent_inliers_at_025", 0)
            + 1 / self.N * percent_inliers_at_025
        )
        self.tracked_metrics["percent_inliers_at_01"] = (
            self.tracked_metrics.get("percent_inliers_at_01", 0)
            + 1 / self.N * percent_inliers_at_01
        )
        self.tracked_metrics["percent_inliers_at_005"] = (
            self.tracked_metrics.get("percent_inliers_at_005", 0)
            + 1 / self.N * percent_inliers_at_005
        )

    def benchmark(self, detector: Detector):
        self.tracked_metrics = {}

        print("Evaluating percent inliers...")
        for idx, batch in enumerate(tqdm(self.dataloader, mininterval=10.0)):
            batch = to_best_device(batch)
            outputs = detector.detect(batch, num_keypoints=self.num_keypoints)
            keypoints_A, keypoints_B = outputs["keypoints"].chunk(2)
            if isinstance(outputs["keypoints"], (tuple, list)):
                keypoints_A, keypoints_B = (
                    torch.stack(keypoints_A),
                    torch.stack(keypoints_B),
                )
            outputs = {"keypoints_A": keypoints_A, "keypoints_B": keypoints_B}
            self.compute_batch_metrics(outputs, batch)
        [
            print(name, metric.item() * self.N / (idx + 1))
            for name, metric in self.tracked_metrics.items()
            if "percent" in name
        ]
        return self.tracked_metrics
