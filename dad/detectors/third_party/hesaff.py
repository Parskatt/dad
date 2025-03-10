from PIL import Image

import torch

from dad.utils import get_best_device
from dad.types import Detector


class HesAff(Detector):
    def __init__(self):
        raise NotImplementedError("Buggy implementation, don't use.")
        super().__init__()
        import pyhesaff

        self.params = pyhesaff.get_hesaff_default_params()

    @property
    def topleft(self):
        return 0.0

    def load_image(self, im_path):
        # pyhesaff doesn't seem to have a decoupled image loading and detection stage
        # so load_image here is just identity
        return {"image": im_path}

    def detect(self, batch, *, num_keypoints, return_dense_probs=False):
        import pyhesaff

        im_path = batch["image"]
        W, H = Image.open(im_path).size
        detections = pyhesaff.detect_feats(im_path)[0][:num_keypoints]
        kps = detections[..., :2]
        kps_n = self.to_normalized_coords(torch.from_numpy(kps), H, W)[None]
        result = {
            "keypoints": kps_n.to(get_best_device()).float(),
            "keypoint_probs": None,
        }
        if return_dense_probs is not None:
            result["dense_probs"] = None
        return result
