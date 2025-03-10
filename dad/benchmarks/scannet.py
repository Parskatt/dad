import os.path as osp
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dad.types import Detector, Matcher, Benchmark
from dad.utils import (
    compute_pose_error,
    estimate_pose_essential,
    estimate_pose_fundamental,
)


class ScanNetBenchmark(Benchmark):
    def __init__(
        self,
        sample_every: int = 1,
        num_ransac_runs=5,
        data_root: str = "data/scannet",
        num_keypoints: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            num_keypoints=num_keypoints,
            sample_every=sample_every,
            num_ransac_runs=num_ransac_runs,
            thresholds=[5, 10, 20],
        )
        self.sample_every = sample_every
        self.topleft = 0.0
        self._post_init()
        self.model: Literal["fundamental", "essential"]
        self.test_pairs: str
        self.benchmark_name: str

    def _post_init(self):
        # set
        raise NotImplementedError("")

    @torch.no_grad()
    def benchmark(self, matcher: Matcher, detector: Detector):
        tmp = np.load(self.test_pairs)
        pairs, rel_pose = tmp["name"], tmp["rel_pose"]
        tot_e_pose = []
        # pair_inds = np.random.choice(range(len(pairs)), size=len(pairs), replace=False)
        for pairind in tqdm(
            range(0, len(pairs), self.sample_every), smoothing=0.9, mininterval=10
        ):
            scene = pairs[pairind]
            scene_name = f"scene0{scene[0]}_00"
            im_A_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
            im_A = Image.open(im_A_path)
            im_B_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[3]}.jpg",
            )
            im_B = Image.open(im_B_path)
            T_gt = rel_pose[pairind].reshape(3, 4)
            R, t = T_gt[:3, :3], T_gt[:3, 3]
            K = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(
                        osp.join(
                            self.data_root,
                            "scans_test",
                            scene_name,
                            "intrinsic",
                            "intrinsic_color.txt",
                        ),
                        "r",
                    )
                    .read()
                    .split("\n")
                    if r
                ]
            )
            w1, h1 = im_A.size
            w2, h2 = im_B.size
            K1 = K.copy()[:3, :3]
            K2 = K.copy()[:3, :3]
            warp, certainty = matcher.match(im_A_path, im_B_path)
            for num_kps in self.num_keypoints:
                keypoints_A = detector.detect_from_path(
                    im_A_path,
                    num_keypoints=num_kps,
                )["keypoints"][0]
                keypoints_B = detector.detect_from_path(
                    im_B_path,
                    num_keypoints=num_kps,
                )["keypoints"][0]
                matches = matcher.match_keypoints(
                    keypoints_A,
                    keypoints_B,
                    warp,
                    certainty,
                    return_tuple=False,
                )
                kpts1, kpts2 = matcher.to_pixel_coordinates(matches, h1, w1, h2, w2)

                offset = detector.topleft - self.topleft
                kpts1, kpts2 = kpts1 - offset, kpts2 - offset

                for _ in range(self.num_ransac_runs):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    threshold = 2.0
                    if self.model == "essential":
                        R_est, t_est = estimate_pose_essential(
                            kpts1.cpu().numpy(),
                            kpts2.cpu().numpy(),
                            w1,
                            h1,
                            K1,
                            w2,
                            h2,
                            K2,
                            threshold,
                        )
                    elif self.model == "fundamental":
                        R_est, t_est = estimate_pose_fundamental(
                            kpts1.cpu().numpy(),
                            kpts2.cpu().numpy(),
                            w1,
                            h1,
                            K1,
                            w2,
                            h2,
                            K2,
                            threshold,
                        )
                    T1_to_2_est = np.concatenate((R_est, t_est[:, None]), axis=-1)
                    e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                    e_pose = max(e_t, e_R)
                    tot_e_pose.append(e_pose)
        return self.compute_auc(np.array(tot_e_pose))


class ScanNet1500(ScanNetBenchmark):
    def _post_init(self):
        self.test_pairs = osp.join(self.data_root, "test.npz")
        self.benchmark_name = "ScanNet1500"
        self.model = "essential"


class ScanNet1500_F(ScanNetBenchmark):
    def _post_init(self):
        self.test_pairs = osp.join(self.data_root, "test.npz")
        self.benchmark_name = "ScanNet1500_F"
        self.model = "fundamental"
