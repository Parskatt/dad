import os
from typing import Optional

import numpy as np
import poselib
from PIL import Image
from tqdm import tqdm

from dad.types import Detector, Matcher, Benchmark


class HPatchesBenchmark(Benchmark):
    def __init__(
        self,
        data_root="data/hpatches",
        sample_every=1,
        num_ransac_runs=5,
        num_keypoints: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            num_keypoints=num_keypoints,
            sample_every=sample_every,
            num_ransac_runs=num_ransac_runs,
            thresholds=[3, 5, 10],
        )
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(self.data_root, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        self.topleft = 0.0
        self._post_init()
        self.skip_seqs: str
        self.scene_names: list[str]

    def _post_init(self):
        # set self.skip_seqs and self.scene_names here
        raise NotImplementedError()

    def benchmark(self, detector: Detector, matcher: Matcher):
        homog_dists = []
        for seq_idx, seq_name in enumerate(tqdm(self.seq_names[:: self.sample_every])):
            if self.skip_seqs in seq_name:
                # skip illumination seqs
                continue
            im_A_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            for im_idx in list(range(2, 7)):
                im_B_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
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
                        H_pred, res = poselib.estimate_homography(
                            kpts1.cpu().numpy(),
                            kpts2.cpu().numpy(),
                            ransac_opt={
                                "max_reproj_error": threshold,
                            },
                        )
                        corners = np.array(
                            [
                                [0, 0, 1],
                                [0, h1 - 1, 1],
                                [w1 - 1, 0, 1],
                                [w1 - 1, h1 - 1, 1],
                            ]
                        )
                        real_warped_corners = np.dot(corners, np.transpose(H))
                        real_warped_corners = (
                            real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                        )
                        warped_corners = np.dot(corners, np.transpose(H_pred))
                        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                        mean_dist = np.mean(
                            np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                        ) / (min(w2, h2) / 480.0)
                        homog_dists.append(mean_dist)
        return self.compute_auc(np.array(homog_dists))


class HPatchesViewpoint(HPatchesBenchmark):
    def _post_init(self):
        self.skip_seqs = "i_"


class HPatchesIllum(HPatchesBenchmark):
    def _post_init(self):
        self.skip_seqs = "v_"
