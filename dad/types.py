from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Detector(ABC, nn.Module):
    @property
    @abstractmethod
    def topleft(self) -> float:
        pass

    @abstractmethod
    def load_image(im_path: Union[str, Path]) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def detect(
        self, batch: dict[str, torch.Tensor], *, num_keypoints, return_dense_probs=False
    ) -> dict[str, torch.Tensor]:
        pass

    @torch.inference_mode
    def detect_from_path(
        self,
        im_path: Union[str, Path],
        *,
        num_keypoints: int,
        return_dense_probs: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.detect(
            self.load_image(im_path),
            num_keypoints=num_keypoints,
            return_dense_probs=return_dense_probs,
        )

    def to_pixel_coords(
        self, normalized_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        if normalized_coords.shape[-1] != 2:
            raise ValueError(
                f"Expected shape (..., 2), but got {normalized_coords.shape}"
            )
        pixel_coords = torch.stack(
            (
                w * (normalized_coords[..., 0] + 1) / 2,
                h * (normalized_coords[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
        return pixel_coords

    def to_normalized_coords(
        self, pixel_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
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


class Matcher(ABC, nn.Module):
    @abstractmethod
    def match(
        self, im_A_path: Union[str | Path], im_B_path: Union[str | Path]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def match_keypoints(
        self,
        keypoints_A: torch.Tensor,
        keypoints_B: torch.Tensor,
        warp: torch.Tensor,
        certainty: torch.Tensor,
        return_tuple: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def to_pixel_coordinates(
        self, matches: torch.Tensor, h1: int, w1: int, h2: int, w2: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class Benchmark(ABC):
    def __init__(
        self,
        *,
        data_root: str,
        thresholds: list[int],
        sample_every: int = 1,
        num_ransac_runs: int = 5,
        num_keypoints: Optional[list[int] | int] = None,
    ) -> None:
        self.num_keypoints = (
            [512, 1024, 2048, 4096, 8192] if num_keypoints is None else num_keypoints
        )
        if isinstance(self.num_keypoints, int):
            self.num_keypoints = [self.num_keypoints]
        self.data_root = data_root
        self.sample_every = sample_every
        self.num_ransac_runs = num_ransac_runs
        self.thresholds = thresholds

    @abstractmethod
    def benchmark(self, *, matcher: Matcher, detector: Detector) -> dict[str, float]:
        pass

    def pose_auc(self, errors):
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in self.thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapz(r, x=e).item() / t)
        return aucs

    def compute_auc(self, errors: np.ndarray) -> dict[str, float]:
        # errors.shape = (len(benchmark)*num_keypoints*num_ransac_runs,)
        errors = (
            errors.reshape((-1, len(self.num_keypoints), self.num_ransac_runs))
            .transpose(0, 2, 1)
            .reshape(-1, len(self.num_keypoints))
        )
        results: dict[str, float] = {}
        for idx in range(len(self.num_keypoints)):
            aucs = self.pose_auc(errors[:, idx])
            for auc, th in zip(aucs, self.thresholds):
                key = (
                    f"{type(self).__name__}_auc_{th}_num_kps_{self.num_keypoints[idx]}"
                )
                results[key] = auc
        return results

    def __call__(self, *, matcher: Matcher, detector: Detector) -> dict[str, float]:
        return self.benchmark(
            matcher=matcher,
            detector=detector,
        )
