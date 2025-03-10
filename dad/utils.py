import math
import warnings
from pathlib import Path
from typing import Optional
from dad.types import Benchmark, Detector, Matcher
import torch.nn as nn
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_best_device(verbose=False):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Fastest device found is: {device}")
    return device


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0_n = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1_n = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret


# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret


def get_grid(B, H, W, device=get_best_device()):
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def fast_inv_2x2(matrix, eps=1e-10):
    return (
        1
        / (torch.linalg.det(matrix)[..., None, None] + eps)
        * torch.stack(
            (
                matrix[..., 1, 1],
                -matrix[..., 0, 1],
                -matrix[..., 1, 0],
                matrix[..., 0, 0],
            ),
            dim=-1,
        ).reshape(*matrix.shape)
    )


def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


def extract_patches_from_coords(x: torch.Tensor, coords: torch.Tensor, patch_size: int):
    # NOTE: we could also do this by just adding extra coords and grid_sampling more
    # but this is easy, and the results should be similar
    B, H, W = x.shape
    B, N, two = coords.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = F.grid_sample(
        unfolded_x.reshape(B, patch_size**2, H, W),
        coords[:, None],
        mode="bilinear",
        align_corners=False,
    )[:, 0]  # B x K_H * K_W x N
    return patches


def sample_keypoints(
    keypoint_probs: torch.Tensor,
    num_samples=8192,
    device=get_best_device(),
    use_nms=True,
    nms_size=1,
    sample_topk=True,
    increase_coverage=True,
    remove_borders=False,
    return_probs=False,
    coverage_pow=1 / 2,
    coverage_size=51,
    subpixel=False,
    scoremap=None,  # required for subpixel
    subpixel_temp=0.5,
):
    B, H, W = keypoint_probs.shape
    if increase_coverage:
        weights = (
            -(torch.linspace(-2, 2, steps=coverage_size, device=device) ** 2)
        ).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (keypoint_probs[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, coverage_size // 2),
        )
        local_density = F.conv2d(
            local_density_x, weights[..., None], padding=(coverage_size // 2, 0)
        )[:, 0]
        keypoint_probs = keypoint_probs * (local_density + 1e-8) ** (-coverage_pow)
    grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)
    if use_nms:
        keypoint_probs = keypoint_probs * (
            keypoint_probs
            == F.max_pool2d(keypoint_probs, nms_size, stride=1, padding=nms_size // 2)
        )
    if remove_borders:
        frame = torch.zeros_like(keypoint_probs)
        # we hardcode 4px, could do it nicer, but whatever
        frame[..., 4:-4, 4:-4] = 1
        keypoint_probs = keypoint_probs * frame
    if sample_topk:
        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(
            keypoint_probs.reshape(B, H * W), num_samples=num_samples, replacement=False
        )
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))
    if subpixel:
        offsets = get_grid(B, nms_size, nms_size).reshape(
            B, nms_size**2, 2
        )  # B x K_H x K_W x 2
        offsets[..., 0] = offsets[..., 0] * nms_size / W
        offsets[..., 1] = offsets[..., 1] * nms_size / H
        keypoint_patch_scores = extract_patches_from_inds(scoremap, inds, nms_size)
        keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(
            dim=1
        )  # B x K_H * K_W x N
        keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
        kps = kps + keypoint_offsets
    if return_probs:
        return kps, torch.gather(keypoint_probs.reshape(B, H * W), dim=1, index=inds)
    return kps


def get_gt_warp(
    depth1,
    depth2,
    T_1to2,
    K1,
    K2,
    depth_interpolation_mode="bilinear",
    relative_depth_error_threshold=0.05,
    H=None,
    W=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if H is None:
        B, H, W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=depth1.device)
                for n in (B, H, W)
            ],
            indexing="ij",
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode=depth_interpolation_mode,
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return torch.cat((x1_n.reshape(B, H, W, 2), x2), dim=-1), prob


def unnormalize_coords(x_n, h, w):
    x = torch.stack(
        (w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def normalize_coords(x, h, w):
    x = torch.stack(
        (2 * (x[..., 0] / w) - 1, 2 * (x[..., 1] / h) - 1), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


@torch.no_grad()
def warp_kpts(
    kpts0,
    depth0,
    depth1,
    T_0to1,
    K0,
    K1,
    smooth_mask=False,
    return_relative_depth_error=False,
    depth_interpolation_mode="bilinear",
    relative_depth_error_threshold=0.05,
):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(
            kpts0,
            depth0,
            depth1,
            T_0to1,
            K0,
            K1,
            smooth_mask=smooth_mask,
            return_relative_depth_error=return_relative_depth_error,
            depth_interpolation_mode="bilinear",
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        valid_nearest, warp_nearest = warp_kpts(
            kpts0,
            depth0,
            depth1,
            T_0to1,
            K0,
            K1,
            smooth_mask=smooth_mask,
            return_relative_depth_error=return_relative_depth_error,
            depth_interpolation_mode="nearest-exact",
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest)
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[
            nearest_valid_bilinear_invalid
        ]
        valid = valid_bilinear | valid_nearest
        return valid, warp

    kpts0_depth = F.grid_sample(
        depth0[:, None],
        kpts0[:, :, None],
        mode=depth_interpolation_mode,
        align_corners=False,
    )[:, 0, :, 0]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None],
        w_kpts0[:, :, None],
        mode=depth_interpolation_mode,
        align_corners=False,
    )[:, 0, :, 0]

    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error / smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0


imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def imgnet_unnormalize(x: torch.Tensor) -> torch.Tensor:
    return x * (imagenet_std[:, None, None].to(x.device)) + (
        imagenet_mean[:, None, None].to(x.device)
    )


def imgnet_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - imagenet_mean[:, None, None].to(x.device)) / (
        imagenet_std[:, None, None].to(x.device)
    )


def tensor_to_pil(x, unnormalize=False, autoscale=False):
    if unnormalize:
        x = imgnet_unnormalize(x)
    if autoscale:
        if x.max() == x.min():
            warnings.warn("x max == x min, cant autoscale")
        else:
            x = (x - x.min()) / (x.max() - x.min())

    x = x.detach()
    if len(x.shape) > 2:
        x = x.permute(1, 2, 0)
    x = x.cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_best_device(batch, device=get_best_device()):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def to_pixel_coords(normalized_coords, h, w) -> torch.Tensor:
    if normalized_coords.shape[-1] != 2:
        raise ValueError(f"Expected shape (..., 2), but got {normalized_coords.shape}")
    pixel_coords = torch.stack(
        (
            w * (normalized_coords[..., 0] + 1) / 2,
            h * (normalized_coords[..., 1] + 1) / 2,
        ),
        axis=-1,
    )
    return pixel_coords


def to_normalized_coords(pixel_coords, h, w) -> torch.Tensor:
    if pixel_coords.shape[-1] != 2:
        raise ValueError(f"Expected shape (..., 2), but got {pixel_coords.shape}")
    normalized_coords = torch.stack(
        (
            2 * (pixel_coords[..., 0]) / w - 1,
            2 * (pixel_coords[..., 1]) / h - 1,
        ),
        axis=-1,
    )
    return normalized_coords


def warp_to_pixel_coords(warp, h1, w1, h2, w2):
    warp1 = warp[..., :2]
    warp1 = torch.stack(
        (
            w1 * (warp1[..., 0] + 1) / 2,
            h1 * (warp1[..., 1] + 1) / 2,
        ),
        axis=-1,
    )
    warp2 = warp[..., 2:]
    warp2 = torch.stack(
        (
            w2 * (warp2[..., 0] + 1) / 2,
            h2 * (warp2[..., 1] + 1) / 2,
        ),
        axis=-1,
    )
    return torch.cat((warp1, warp2), dim=-1)


def to_homogeneous(x):
    ones = torch.ones_like(x[..., -1:])
    return torch.cat((x, ones), dim=-1)


to_hom = to_homogeneous  # alias


def from_homogeneous(xh, eps=1e-12):
    return xh[..., :-1] / (xh[..., -1:] + eps)


from_hom = from_homogeneous  # alias


def homog_transform(Homog, x):
    xh = to_homogeneous(x)
    yh = (Homog @ xh.mT).mT
    y = from_homogeneous(yh)
    return y


def get_homog_warp(Homog, H, W, device=get_best_device()):
    grid = torch.meshgrid(
        torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
        torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
        indexing="ij",
    )

    x_A = torch.stack((grid[1], grid[0]), dim=-1)[None]
    x_A_to_B = homog_transform(Homog, x_A)
    mask = ((x_A_to_B > -1) * (x_A_to_B < 1)).prod(dim=-1).float()
    return torch.cat((x_A.expand(*x_A_to_B.shape), x_A_to_B), dim=-1), mask


def dual_log_softmax_matcher(desc_A, desc_B, inv_temperature=1, normalize=False):
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=-1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=-1, keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    logP = corr.log_softmax(dim=-2) + corr.log_softmax(dim=-1)
    return logP


def dual_softmax_matcher(desc_A, desc_B, inv_temperature=1, normalize=False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=-1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=-1, keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim=-2) * corr.softmax(dim=-1)
    return P


def conditional_softmax_matcher(desc_A, desc_B, inv_temperature=1, normalize=False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A / desc_A.norm(dim=-1, keepdim=True)
        desc_B = desc_B / desc_B.norm(dim=-1, keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P_B_cond_A = corr.softmax(dim=-1)
    P_A_cond_B = corr.softmax(dim=-2)

    return P_A_cond_B, P_B_cond_A


def draw_kpts(im, kpts, radius=2, width=1):
    im = np.array(im)
    # Convert keypoints to numpy array
    kpts_np = kpts.cpu().numpy()

    # Create a copy of the image to draw on
    ret = im.copy()

    # Define green color (BGR format in OpenCV)
    green_color = (0, 255, 0)

    # Draw green plus signs for each keypoint
    for x, y in kpts_np:
        # Convert to integer coordinates
        x, y = int(x), int(y)

        # Draw horizontal line of the plus sign
        cv2.line(ret, (x - radius, y), (x + radius, y), green_color, width)
        # Draw vertical line of the plus sign
        cv2.line(ret, (x, y - radius), (x, y + radius), green_color, width)

    return ret


def masked_log_softmax(logits, mask):
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[mask] = logits[mask]
    log_p = masked_logits.log_softmax(dim=-1)
    return log_p


def masked_softmax(logits, mask):
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[mask] = logits[mask]
    log_p = masked_logits.softmax(dim=-1)
    return log_p


def kde(x, std=0.1, half=True, down=None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half()  # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-(torch.cdist(x, x[::down]) ** 2) / (2 * std**2)).exp()
    else:
        scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def midpoint_triangulation_unbatched(v1s_local, v2s_local, T1, T2, return_angles=False):
    R1 = T1[:3, :3]  # 3x3 rotation matrix
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]  # 3x1 translation vector
    t2 = T2[:3, 3]

    # Calculate camera centers (single position for each camera)
    C1 = -torch.matmul(R1.T, t1)  # (3,)
    C2 = -torch.matmul(R2.T, t2)  # (3,)

    # # Transform view vectors from local to world coordinates
    # # World vector = R * local_vector

    v1s_world = F.normalize(v1s_local @ R1)  # (N x 3)
    v2s_world = F.normalize(v2s_local @ R2)  # (N x 3)

    # # Vector between camera centers (broadcast to match number of points)
    b = C2 - C1  # (3,)
    num_points = v1s_local.shape[0]
    bs = b.unsqueeze(0).expand(num_points, -1)  # (N x 3)

    # Compute direction vectors between closest points on rays
    cross1 = torch.cross(v1s_world, v2s_world)  # N x 3
    cross2 = torch.cross(bs, v2s_world)  # N x 3

    # Calculate parameters using cross products
    s = torch.sum(cross2 * cross1, dim=1) / torch.sum(cross1 * cross1, dim=1)
    t = torch.sum(torch.cross(bs, v1s_world) * cross1, dim=1) / torch.sum(
        cross1 * cross1, dim=1
    )

    # Find points on each ray in world coordinates
    P1s = C1.unsqueeze(0) + s.unsqueeze(1) * v1s_world  # (N x 3)
    P2s = C2.unsqueeze(0) + t.unsqueeze(1) * v2s_world  # (N x 3)

    # For parallel rays, use camera midpoints
    # midpoint = (C1 + C2) / 2
    # midpoints = midpoint.unsqueeze(0).expand(num_points, -1)
    midpoint = (P1s + P2s) / 2
    if not return_angles:
        return midpoint
    tri_angles = (
        180 / torch.pi * torch.acos((v1s_world * v2s_world).sum(dim=1).clip(0, 1.0))
    )
    return midpoint, tri_angles


def midpoint_triangulation(
    x_A: torch.Tensor,
    x_B: torch.Tensor,
    T_A: torch.Tensor,
    T_B: torch.Tensor,
    return_angles=False,
):
    batch, num_points, three = x_A.shape
    assert three == 3
    # rotation matrix
    R_A = T_A[..., :3, :3]  # (B x 3 x 3)
    R_B = T_B[..., :3, :3]
    # translation vector
    t_A = T_A[..., :3, 3]  # (B x 3)
    t_B = T_B[..., :3, 3]

    # Calculate camera centers (single position for each camera)
    C_A = (R_A.mT @ -t_A[..., None])[..., 0]  # (B x 3 x 3) * (B x 3 x 1) -> (B x 3)
    C_B = (R_B.mT @ -t_B[..., None])[..., 0]  # (B x 3 x 3) * (B x 3 x 1) -> (B x 3)

    # # Transform view vectors from local to world coordinates
    # # World vector = R * local_vector
    ray_A_world = F.normalize(x_A @ R_A, dim=-1)  # (B x N x 3)
    ray_B_world = F.normalize(x_B @ R_B, dim=-1)  # (B x N x 3)

    # # Vector between camera centers (broadcast to match number of points)
    b = C_B - C_A  # (B x 3 x 1)
    bs = b.reshape(batch, 1, three).expand(batch, num_points, three)  # (B x N x 3)

    # Compute direction vectors between closest points on rays
    cross1 = torch.linalg.cross(ray_A_world, ray_B_world)  # B x N x 3
    cross2 = torch.linalg.cross(bs, ray_B_world)  # B x N x 3
    cross3 = torch.linalg.cross(bs, ray_A_world)  # B x N x 3

    # Calculate parameters using cross products
    denom = torch.sum(cross1 * cross1, dim=-1)  # (B x N x 3) -> (B x N)
    s = torch.sum(cross2 * cross1, dim=-1) / denom  # B x N
    t = torch.sum(cross3 * cross1, dim=-1) / denom  # B x N

    # Find points on each ray in world coordinates
    P_A = (
        C_A[:, None] + s[..., None] * ray_A_world
    )  # (B x 1 x 3), (B x N x 1), (B x N x 3) -> (B, N, 3)
    P_B = (
        C_B[:, None] + t[..., None] * ray_B_world
    )  # (B x 1 x 3), (B x N x 1), (B x N x 3) -> (B, N, 3)

    # For parallel rays, use camera midpoints
    midpoint = (P_A + P_B) / 2  # (B x N x 3)
    if not return_angles:
        return midpoint
    tri_angles = (
        180
        / torch.pi
        * torch.acos((ray_A_world * ray_B_world).sum(dim=-1).clip(0, 1.0))
    )  # B x N
    return midpoint, tri_angles


class SkillIssue(NotImplementedError):
    pass


def calibrate(x: torch.Tensor, K: torch.Tensor):
    # x: ..., 2
    # K: ..., 3, 3
    return to_homogeneous(x) @ K.inverse().mT


def project(X: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
    # X: ..., 3
    # T: ..., 4, 4
    # K: ..., 3, 3
    return from_homogeneous(from_homogeneous(to_homogeneous(X) @ T.mT) @ K.mT)


def eye_like(x):
    C, D = x.shape[-2:]
    if C != D:
        raise ValueError(f"Shape not square: {x.shape}")
    e = torch.eye(D).to(x).expand_as(x)
    return e


def triangulate(x_A, x_B, T_A_to_B, K_A, K_B, method="midpoint", return_angles=False):
    if method != "midpoint":
        raise SkillIssue("You should use midpoint instead")
    T_B = T_A_to_B
    T_A = eye_like(T_B)
    x_A_calib = calibrate(x_A, K_A)
    x_B_calib = calibrate(x_B, K_B)
    result = midpoint_triangulation(
        x_A_calib, x_B_calib, T_A, T_B, return_angles=return_angles
    )
    return result


def visualize_keypoints(img_path, vis_path, detector: Detector, num_keypoints: int):
    img_path, vis_path = Path(img_path), Path(vis_path).with_suffix(".png")
    img = Image.open(img_path)
    detections = detector.detect_from_path(
        img_path, num_keypoints=num_keypoints, return_dense_probs=True
    )
    W, H = img.size
    kps = detections["keypoints"]
    kps = detector.to_pixel_coords(kps, H, W)
    (vis_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_kpts(img, kps[0])).save(vis_path)
    if detections["dense_probs"] is not None:
        tensor_to_pil(detections["dense_probs"].squeeze().cpu(), autoscale=True).save(
            vis_path.as_posix().replace(".png", "_dense_probs.png")
        )


def run_qualitative_examples(
    *, model: Detector, workspace_path: str | Path, test_num_keypoints
):
    import dad

    workspace_path = Path(workspace_path)
    torch.cuda.empty_cache()
    for im_path in [
        "assets/0015_A.jpg",
        "assets/0015_B.jpg",
        "assets/0032_A.jpg",
        "assets/0032_B.jpg",
        "assets/apprentices.jpg",
        "assets/rectangles_and_circles.png",
    ]:
        visualize_keypoints(
            im_path, 
            workspace_path / "vis" / str(dad.GLOBAL_STEP) / im_path,
            model,
            num_keypoints=test_num_keypoints,
        )
    torch.cuda.empty_cache()


def get_experiment_name(experiment_file: str):
    return (
        Path(experiment_file)
        .relative_to(Path("experiments").absolute())
        .with_suffix("")
        .as_posix()
    )


def get_data_iterator(dataset, sample_weights, batch_size, num_steps):
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=batch_size * num_steps, replacement=False
    )
    return iter(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=batch_size,
        )
    )


def run_benchmarks(
    benchmarks: list[Benchmark],
    matcher: Matcher,
    detector: Detector,
    *,
    step: int,
    num_keypoints: Optional[list[int] | int] = None,
    sample_every: Optional[int] = 1,
):
    import wandb

    torch.cuda.empty_cache()
    if isinstance(num_keypoints, int):
        num_keypoints = [num_keypoints]

    for bench in benchmarks:
        wandb.log(
            bench(num_keypoints=num_keypoints, sample_every=sample_every)(
                matcher=matcher,
                detector=detector,
            ),
            step=step,
        )
    torch.cuda.empty_cache()


def estimate_pose_essential(
    kps_A: np.ndarray,
    kps_B: np.ndarray,
    w_A: int,
    h_A: int,
    K_A: np.ndarray,
    w_B: int,
    h_B: int,
    K_B: np.ndarray,
    th: float,
) -> tuple[np.ndarray, np.ndarray]:
    import poselib

    camera1 = {
        "model": "PINHOLE",
        "width": w_A,
        "height": h_A,
        "params": K_A[[0, 1, 0, 1], [0, 1, 2, 2]],
    }
    camera2 = {
        "model": "PINHOLE",
        "width": w_B,
        "height": h_B,
        "params": K_B[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    pose, res = poselib.estimate_relative_pose(
        kps_A,
        kps_B,
        camera1,
        camera2,
        ransac_opt={
            "max_epipolar_error": th,
        },
    )
    return pose.R, pose.t


def poselib_fundamental(x1, x2, opt):
    import poselib

    F, info = poselib.estimate_fundamental(x1, x2, opt, {})
    inl = info["inliers"]
    return F, inl


def estimate_pose_fundamental(
    kps_A: np.ndarray,
    kps_B: np.ndarray,
    w_A: int,
    h_A: int,
    K_A: np.ndarray,
    w_B: int,
    h_B: int,
    K_B: np.ndarray,
    th: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(kps_A) < 8:
        return np.eye(3), np.zeros(3)
    F, inl = poselib_fundamental(
        kps_A,
        kps_B,
        opt={
            "max_epipolar_error": th,
        },
    )
    E: np.ndarray = K_B.T @ F @ K_A
    kps_calib_A = from_hom(
        calibrate(torch.from_numpy(kps_A).float(), torch.from_numpy(K_A).float())
    ).numpy()
    kps_calib_B = from_hom(
        calibrate(torch.from_numpy(kps_B).float(), torch.from_numpy(K_B).float())
    ).numpy()
    E = E.astype(np.float64)
    _, R, t, good = cv2.recoverPose(E, kps_calib_A, kps_calib_B)
    t = t[:, 0]
    return R, t


def so2(radians):
    return torch.tensor(
        [
            [math.cos(radians), math.sin(radians), 0],
            [-math.sin(radians), math.cos(radians), 0],
            [0, 0, 1.0],
        ]
    )


def rotate_normalized_points(points: torch.Tensor, angle: float):
    # points are between -1, 1, Nx2
    # angle is float [0, 360]
    radians = angle * math.pi / 180
    rot_mat = so2(radians).to(points)
    return points @ rot_mat[:2, :2].T


def compute_detector_correlation(dets1: torch.Tensor, dets2: torch.Tensor, th: float):
    # det1.shape = (K, 2)
    # K = num keypoints
    d = torch.cdist(dets1, dets2, compute_mode="donot_use_mm_for_euclid_dist")
    d12 = d.amin(dim=1)
    d21 = d.amin(dim=0)
    mnn = (d == d12) * (d == d21)
    corr = mnn.float()
    corr[d > th] = 0.0
    return corr.sum(dim=1).mean(), corr.sum(dim=0).mean()


def cross_entropy(log_p_hat: torch.Tensor, p: torch.Tensor):
    return -(log_p_hat * p).sum(dim=-1)


def kl_div(p: torch.Tensor, log_p_hat: torch.Tensor):
    return cross_entropy(log_p_hat, p) - cross_entropy((p + 1e-12).log(), p)


def generalized_mean(r, p1, p2):
    return (1 / 2 * (p1**r + p2**r)) ** (1 / r)


def setup_experiment(experiment_file, root_workspace_path="workspace", disable_wandb=False):
    import wandb

    experiment_name = get_experiment_name(experiment_file)
    wandb.init(
        project="dad",
        mode="online" if not disable_wandb else "disabled",
        name=experiment_name.replace("/", "-"),
    )
    workspace_path = Path(root_workspace_path) / experiment_name
    workspace_path.mkdir(parents=True, exist_ok=True)
    return workspace_path


def check_not_i16(im):
    if im.mode == "I;16":
        raise NotImplementedError("Can't handle 16 bit images")

def wrap_in_sbatch(command, account, time_alloc = "2-23:00:00"):
    sbatch_command = f"""#!/bin/bash
#SBATCH -A {account}
#SBATCH -t {time_alloc}
#SBATCH -o %j.out
#SBATCH --gpus 1
#SBATCH --nodes 1

# Job script commands follow
# Print some GPU info" \
source .venv/bin/activate
{command}
"""
    return sbatch_command