import numpy as np
import torch

from dad.types import Detector
import cv2

from dad.utils import get_best_device


class HarrisAff(Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(
            numOctaves=6, corn_thresh=0.0, DOG_thresh=0.0, maxCorners=8192, num_layers=4
        )

    @property
    def topleft(self):
        return 0.0

    def load_image(self, im_path):
        return {"image": cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)}

    @torch.inference_mode()
    def detect(self, batch, *, num_keypoints, return_dense_probs=False) -> dict[str, torch.Tensor]:
        img = batch["image"]
        H, W = img.shape
        # Detect keypoints
        kps = self.detector.detect(img)
        kps = np.array([kp.pt for kp in kps])[:num_keypoints]
        kps_n = self.to_normalized_coords(torch.from_numpy(kps), H, W)[None]
        detections = {"keypoints": kps_n.to(get_best_device()).float(), "keypoint_probs": None}
        if return_dense_probs:
            detections["dense_probs"] = None
        return detections
