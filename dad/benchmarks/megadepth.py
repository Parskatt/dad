from typing import Literal, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from dad.types import Detector, Matcher, Benchmark
from dad.utils import (
    compute_pose_error,
    compute_relative_pose,
    estimate_pose_essential,
    estimate_pose_fundamental,
)


class MegaDepthPoseEstimationBenchmark(Benchmark):
    def __init__(
        self,
        data_root="data/megadepth",
        sample_every=1,
        num_ransac_runs=5,
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
        self.topleft = 0.5
        self._post_init()
        self.model: Literal["fundamental", "essential"]
        self.scene_names: list[str]
        self.benchmark_name: str

    def _post_init(self):
        raise NotImplementedError(
            "Add scene names and benchmark name in derived class _post_init"
        )

    def benchmark(
        self,
        detector: Detector,
        matcher: Matcher,
    ):
        self.scenes = [
            np.load(f"{self.data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]

        data_root = self.data_root
        tot_e_pose = []
        n_matches = []
        for scene_ind in range(len(self.scenes)):
            scene = self.scenes[scene_ind]
            pairs = scene["pair_infos"]
            intrinsics = scene["intrinsics"]
            poses = scene["poses"]
            im_paths = scene["image_paths"]
            pair_inds = range(len(pairs))
            for pairind in (
                pbar := tqdm(
                    pair_inds[:: self.sample_every],
                    desc="Current AUC: ?",
                    mininterval=10,
                )
            ):
                idx1, idx2 = pairs[pairind][0]
                K1 = intrinsics[idx1].copy()
                T1 = poses[idx1].copy()
                R1, t1 = T1[:3, :3], T1[:3, 3]
                K2 = intrinsics[idx2].copy()
                T2 = poses[idx2].copy()
                R2, t2 = T2[:3, :3], T2[:3, 3]
                R, t = compute_relative_pose(R1, t1, R2, t2)
                im_A_path = f"{data_root}/{im_paths[idx1]}"
                im_B_path = f"{data_root}/{im_paths[idx2]}"

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
                    n_matches.append(matches.shape[0])
                    im_A = Image.open(im_A_path)
                    w1, h1 = im_A.size
                    im_B = Image.open(im_B_path)
                    w2, h2 = im_B.size
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
                pbar.set_description(
                    f"Current AUCS: {self.compute_auc(np.array(tot_e_pose))}"
                )
        n_matches = np.array(n_matches)
        print(n_matches.mean(), np.median(n_matches), np.std(n_matches))
        return self.compute_auc(np.array(tot_e_pose))


class Mega1500(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        self.benchmark_name = "Mega1500"
        self.model = "essential"


class Mega1500_F(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        # self.benchmark_name = "Mega1500_F"
        self.model = "fundamental"


class MegaIMCPT(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "mega_8_scenes_0008_0.1_0.3.npz",
            "mega_8_scenes_0008_0.3_0.5.npz",
            "mega_8_scenes_0019_0.1_0.3.npz",
            "mega_8_scenes_0019_0.3_0.5.npz",
            "mega_8_scenes_0021_0.1_0.3.npz",
            "mega_8_scenes_0021_0.3_0.5.npz",
            "mega_8_scenes_0024_0.1_0.3.npz",
            "mega_8_scenes_0024_0.3_0.5.npz",
            "mega_8_scenes_0025_0.1_0.3.npz",
            "mega_8_scenes_0025_0.3_0.5.npz",
            "mega_8_scenes_0032_0.1_0.3.npz",
            "mega_8_scenes_0032_0.3_0.5.npz",
            "mega_8_scenes_0063_0.1_0.3.npz",
            "mega_8_scenes_0063_0.3_0.5.npz",
            "mega_8_scenes_1589_0.1_0.3.npz",
            "mega_8_scenes_1589_0.3_0.5.npz",
        ]
        # self.benchmark_name = "MegaIMCPT"
        self.model = "essential"


class MegaIMCPT_F(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "mega_8_scenes_0008_0.1_0.3.npz",
            "mega_8_scenes_0008_0.3_0.5.npz",
            "mega_8_scenes_0019_0.1_0.3.npz",
            "mega_8_scenes_0019_0.3_0.5.npz",
            "mega_8_scenes_0021_0.1_0.3.npz",
            "mega_8_scenes_0021_0.3_0.5.npz",
            "mega_8_scenes_0024_0.1_0.3.npz",
            "mega_8_scenes_0024_0.3_0.5.npz",
            "mega_8_scenes_0025_0.1_0.3.npz",
            "mega_8_scenes_0025_0.3_0.5.npz",
            "mega_8_scenes_0032_0.1_0.3.npz",
            "mega_8_scenes_0032_0.3_0.5.npz",
            "mega_8_scenes_0063_0.1_0.3.npz",
            "mega_8_scenes_0063_0.3_0.5.npz",
            "mega_8_scenes_1589_0.1_0.3.npz",
            "mega_8_scenes_1589_0.3_0.5.npz",
        ]
        # self.benchmark_name = "MegaIMCPT_F"
        self.model = "fundamental"
