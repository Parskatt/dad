from typing import Callable
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

import dad
from dad.utils import (
    get_gt_warp,
    masked_log_softmax,
    sample_keypoints,
    kl_div,
)


class RLLoss(nn.Module):
    def __init__(
        self,
        *,
        reward_function: Callable[[torch.Tensor], torch.Tensor],
        smoothing_size: int,
        sampling_kde_size: int,
        nms_size: int,
        num_sparse: int,
        regularization_loss_weight: float,
        coverage_pow: float,
        topk: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        X = torch.linspace(-1, 1, smoothing_size, device=device)
        G = (-(X**2) / (2 * 1 / 2**2)).exp()
        G = G / G.sum()
        self.smoothing_kernel = G[None, None, None, :]
        self.smoothing_size = smoothing_size
        self.regularization_loss_weight = regularization_loss_weight
        self.nms_size = nms_size
        self.num_sparse = num_sparse
        self.reward_function = reward_function
        self.sampling_kde_size = sampling_kde_size
        self.coverage_pow = coverage_pow
        self.topk = topk

    def compute_matchability(self, keypoint_p, has_depth, B, K, H, W, device="cuda"):
        smooth_keypoint_p = F.conv2d(
            keypoint_p.reshape(B, 1, H, W),
            weight=self.smoothing_kernel,
            padding=(self.smoothing_size // 2, 0),
        )
        smooth_keypoint_p = F.conv2d(
            smooth_keypoint_p,
            weight=self.smoothing_kernel.mT,
            padding=(0, self.smoothing_size // 2),
        )
        log_p_hat = (
            (smooth_keypoint_p + 1e-8).log().reshape(B, H * W).log_softmax(dim=-1)
        )
        smooth_has_depth = F.conv2d(
            has_depth.reshape(B, 1, H, W),
            weight=self.smoothing_kernel,
            padding=(0, self.smoothing_size // 2),
        )
        smooth_has_depth = F.conv2d(
            smooth_has_depth,
            weight=self.smoothing_kernel.mT,
            padding=(self.smoothing_size // 2, 0),
        ).reshape(B, H * W)
        p = smooth_has_depth / smooth_has_depth.sum(dim=-1, keepdim=True)
        return kl_div(p, log_p_hat)

    def compute_loss(self, batch, model):
        outputs = model(batch)
        keypoint_logits_A, keypoint_logits_B = outputs["scoremap"].chunk(2)
        B, K, H, W = keypoint_logits_A.shape

        gt_warp_A_to_B, valid_mask_A_to_B = get_gt_warp(
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=H,
            W=W,
        )
        gt_warp_B_to_A, valid_mask_B_to_A = get_gt_warp(
            batch["im_B_depth"],
            batch["im_A_depth"],
            batch["T_1to2"].inverse(),
            batch["K2"],
            batch["K1"],
            H=H,
            W=W,
        )
        keypoint_logits_A = keypoint_logits_A.reshape(B, K, H * W)
        keypoint_logits_B = keypoint_logits_B.reshape(B, K, H * W)
        keypoint_logits = torch.cat((keypoint_logits_A, keypoint_logits_B))

        B = 2 * B
        gt_warp = torch.cat((gt_warp_A_to_B, gt_warp_B_to_A))
        valid_mask = torch.cat((valid_mask_A_to_B, valid_mask_B_to_A))
        valid_mask = valid_mask.reshape(B, H * W)
        keypoint_logits_backwarped = F.grid_sample(
            torch.cat((keypoint_logits_B, keypoint_logits_A)).reshape(B, K, H, W),
            gt_warp[..., -2:].reshape(B, H, W, 2).float(),
            align_corners=False,
            mode="bicubic",
        )

        keypoint_logits_backwarped = (keypoint_logits_backwarped).reshape(B, K, H * W)

        depth = F.interpolate(
            torch.cat(
                (batch["im_A_depth"][:, None], batch["im_B_depth"][:, None]), dim=0
            ),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        has_depth = (depth > 0).float().reshape(B, H * W)
        keypoint_p = (
            keypoint_logits.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
        )
        matchability_loss = self.compute_matchability(
            keypoint_p, has_depth, B, K, H, W
        ).mean()
        B = B // 2
        M = self.num_sparse
        torch.set_grad_enabled(False)
        kpts_A = sample_keypoints(
            keypoint_p[:B].reshape(B, H, W),
            use_nms=True,
            nms_size=self.nms_size,
            sample_topk=self.topk,
            num_samples=M,
            coverage_size=self.sampling_kde_size,
            increase_coverage=True,
            coverage_pow=self.coverage_pow,
            subpixel=False,
            scoremap=keypoint_logits[:B].reshape(B, H, W),
        )
        kpts_B = sample_keypoints(
            keypoint_p[B:].reshape(B, H, W),
            use_nms=True,
            nms_size=self.nms_size,
            sample_topk=self.topk,
            num_samples=M,
            coverage_size=self.sampling_kde_size,
            increase_coverage=True,
            coverage_pow=self.coverage_pow,
            subpixel=False,
            scoremap=keypoint_logits[B:].reshape(B, H, W),
        )
        kpts_A_to_B = F.grid_sample(
            gt_warp_A_to_B[..., 2:].float().permute(0, 3, 1, 2),
            kpts_A[..., None, :],
            align_corners=False,
            mode="bilinear",
        )[..., 0].mT
        legit_A_to_B = (
            F.grid_sample(
                valid_mask_A_to_B.reshape(B, 1, H, W),
                kpts_A[..., None, :],
                align_corners=False,
                mode="bilinear",
            )[..., 0, :, 0]
            > 0
        )
        kpts_B_to_A = F.grid_sample(
            gt_warp_B_to_A[..., 2:].float().permute(0, 3, 1, 2),
            kpts_B[..., None, :],
            align_corners=False,
            mode="bilinear",
        )[..., 0].mT
        legit_B_to_A = (
            F.grid_sample(
                valid_mask_B_to_A.reshape(B, 1, H, W),
                kpts_B[..., None, :],
                align_corners=False,
                mode="bilinear",
            )[..., 0, :, 0]
            > 0
        )
        D_A_to_B = torch.cdist(kpts_A_to_B, kpts_B)
        D_B_to_A = torch.cdist(kpts_B_to_A, kpts_A)

        min_dist_A_to_B = D_A_to_B.amin(dim=-1)
        min_dist_B_to_A = D_B_to_A.amin(dim=-1)
        torch.set_grad_enabled(True)

        inlier_threshold = 0.005
        inliers_A_to_B = min_dist_A_to_B < inlier_threshold
        percent_inliers_A_to_B = inliers_A_to_B[legit_A_to_B].float().mean()
        wandb.log(
            {"mega_percent_inliers": percent_inliers_A_to_B.item()},
            step=dad.GLOBAL_STEP,
        )

        reward_A_to_B = self.reward_function(min_dist_A_to_B)
        reward_B_to_A = self.reward_function(min_dist_B_to_A)
        sparse_kpt_logits_A = F.grid_sample(
            keypoint_logits_A.reshape(B, 1, H, W),
            kpts_A[:, None].detach(),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, M)
        sparse_kpt_logits_B = F.grid_sample(
            keypoint_logits_B.reshape(B, 1, H, W),
            kpts_B[:, None].detach(),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, M)
        sparse_kpt_log_p_A = masked_log_softmax(sparse_kpt_logits_A, legit_A_to_B)
        sparse_kpt_log_p_B = masked_log_softmax(sparse_kpt_logits_B, legit_B_to_A)

        tot_loss = 0.0
        sparse_loss = (
            -(reward_A_to_B[legit_A_to_B] * sparse_kpt_log_p_A[legit_A_to_B]).sum()
            - (reward_B_to_A[legit_B_to_A] * sparse_kpt_log_p_B[legit_B_to_A]).sum()
        )
        tot_loss = tot_loss + sparse_loss
        tot_loss = tot_loss + self.regularization_loss_weight * matchability_loss
        return tot_loss

    def forward(self, batch, model):
        return self.compute_loss(batch, model)


class MaxDistillLoss(nn.Module):
    def __init__(self, *teachers: list[dad.Detector]):
        self.teachers = teachers

    def forward(self, batch, student):
        p_teachers = []
        with torch.inference_mode():
            for teacher in self.teachers:
                scoremap: torch.Tensor = teacher(batch)["scoremap"]
                B, one, H, W = scoremap.shape
                p_teachers.append(
                    scoremap.reshape(B, H * W).softmax(dim=1).reshape(B, 1, H, W)
                )
        p_max = torch.maximum(*p_teachers).clone()
        p_max = p_max / p_max.sum(dim=(-2, -1), keepdim=True)
        scoremap_student = student(batch)
        scoremap: torch.Tensor = scoremap_student["scoremap"]
        B, one, H, W = scoremap.shape
        log_p_model = scoremap.reshape(B, H * W).log_softmax(dim=1).reshape(B, 1, H, W)
        kl = (
            -(p_max * log_p_model).sum() / B + (p_max * (p_max + 1e-10).log()).sum() / B
        )
        wandb.log({"distill_kl": kl.item()}, step=dad.GLOBAL_STEP)
        return kl
