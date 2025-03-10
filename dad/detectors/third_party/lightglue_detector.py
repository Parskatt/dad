from pathlib import Path
from typing import Union
import torch
from .lightglue.utils import load_image
from dad.utils import (
    get_best_device,
)
from dad.types import Detector


class LightGlueDetector(Detector):
    def __init__(self, model, resize=None, **kwargs):
        super().__init__()
        self.model = model(**kwargs).eval().to(get_best_device())
        if resize is not None:
            self.model.preprocess_conf["resize"] = resize

    @property
    def topleft(self):
        return 0.0

    def load_image(self, im_path: Union[str, Path]):
        return {"image": load_image(im_path).to(get_best_device())}

    @torch.inference_mode()
    def detect(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_keypoints: int,
        return_dense_probs: bool = False,
    ):
        image = batch["image"]
        self.model.conf.max_num_keypoints = num_keypoints
        ret = self.model.extract(image)
        kpts = self.to_normalized_coords(
            ret["keypoints"], ret["image_size"][0, 1], ret["image_size"][0, 0]
        )
        result = {"keypoints": kpts, "keypoint_probs": None}
        if return_dense_probs:
            result["dense_probs"] = ret["dense_probs"] if "dense_probs" in ret else None
        return result
