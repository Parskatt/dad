import torch

from .config import get_config
from .model.load_models import load_detector
import cv2
import numpy as np

from . import geometry_tools as geo_tools
from dad.utils import get_best_device

from dad.types import Detector


def upsample_pyramid(image, upsampled_levels, scale_factor_levels):
    ## image np.array([C, H, W]), upsampled_levels int
    up_pyramid = []
    for j in range(upsampled_levels):
        factor = scale_factor_levels ** (upsampled_levels - j)
        up_image = cv2.resize(
            image.transpose(1, 2, 0),
            dsize=(0, 0),
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_LINEAR,
        )
        up_pyramid.append(up_image[np.newaxis])

    return up_pyramid


class MultiScaleFeatureExtractor(Detector):
    def __init__(self, args):
        super().__init__()
        ## configurations
        self.default_num_points = args.num_points
        self.pyramid_levels = args.pyramid_levels
        self.upsampled_levels = args.upsampled_levels
        self.resize = None  # TODO: should be working with args.resize but not sure
        self.border_size = args.border_size
        self.nms_size = args.nms_size
        self.desc_scale_factor = 2.0
        self.scale_factor_levels = np.sqrt(2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_detector(args, device)

        ## points level define (Image Pyramid level)

        self.levels = self.pyramid_levels + self.upsampled_levels + 1
        ## GPU
        self.device = device

    @property
    def topleft(self):
        return 0.0

    def load_image(self, path):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)  ## (1, H, W)
        # Get current dimensions
        h, w = im.shape
        if self.resize is not None:
            # Determine which dimension is longer
            if h > w:
                # Height is longer, calculate new width to maintain aspect ratio
                new_h = self.resize
                new_w = int(w * (self.resize / h))
            else:
                # Width is longer, calculate new height to maintain aspect ratio
                new_w = self.resize
                new_h = int(h * (self.resize / w))
            # Resize the image
            im = cv2.resize(im, (new_w, new_h))
        im = im.astype(float)[np.newaxis, :, :] / im.max()
        return {"image": im}

    @torch.inference_mode()
    def detect(self, batch, *, num_keypoints, return_dense_probs=False):
        image = batch["image"]
        one, H, W = image.shape
        score_maps, ori_maps = self._compute_score_maps(image)
        im_pts = self._estimate_keypoint_coordinates(
            score_maps, num_points=num_keypoints
        )
        pixel_coords = im_pts[..., :2]
        # print(pixel_coords)
        # maybe_scale = im_pts[...,2]
        # maybe_score = im_pts[...,3]
        im_pts_n = (
            self.to_normalized_coords(torch.from_numpy(pixel_coords)[None], H, W)
            .to(get_best_device())
            .float()
        )
        result = {"keypoints": im_pts_n}
        if return_dense_probs:
            result["scoremap"] = None
        return result

    def _compute_score_maps(self, image):
        from skimage.transform import pyramid_gaussian

        pyramid = pyramid_gaussian(
            image, max_layer=self.pyramid_levels, downscale=self.scale_factor_levels
        )
        up_pyramid = upsample_pyramid(
            image,
            upsampled_levels=self.upsampled_levels,
            scale_factor_levels=self.scale_factor_levels,
        )

        score_maps = {}
        ori_maps = {}
        for j, down_image in enumerate(pyramid):  ## Pyramid is downsampling images.
            key_idx = j + 1 + self.upsampled_levels
            score_maps, ori_maps = self._obtain_feature_maps(
                down_image, key_idx, score_maps, ori_maps
            )

        if self.upsampled_levels:
            for j, up_image in enumerate(
                up_pyramid
            ):  ## Upsample levels is for upsampling images.
                key_idx = j + 1
                score_maps, ori_maps = self._obtain_feature_maps(
                    up_image, key_idx, score_maps, ori_maps
                )

        return score_maps, ori_maps

    def _obtain_feature_maps(self, im, key_idx, score_maps, ori_maps):
        im = torch.tensor(im).unsqueeze(0).to(torch.float32).cuda()
        im_scores, ori_map = self.model(im)
        im_scores = geo_tools.remove_borders(
            im_scores[0, 0, :, :].cpu().detach().numpy(), borders=self.border_size
        )

        score_maps["map_" + str(key_idx)] = im_scores
        ori_maps["map_" + str(key_idx)] = ori_map

        return score_maps, ori_maps

    def _estimate_keypoint_coordinates(self, score_maps, num_points=None):
        num_points = num_points if num_points is not None else self.default_num_points
        point_level = []
        tmp = 0.0
        factor_points = self.scale_factor_levels**2
        for idx_level in range(self.levels):
            tmp += factor_points ** (-1 * (idx_level - self.upsampled_levels))
            point_level.append(
                self.default_num_points
                * factor_points ** (-1 * (idx_level - self.upsampled_levels))
            )

        point_level = np.asarray(list(map(lambda x: int(x / tmp) + 1, point_level)))

        im_pts = []
        for idx_level in range(self.levels):
            scale_value = self.scale_factor_levels ** (
                idx_level - self.upsampled_levels
            )
            scale_factor = 1.0 / scale_value

            h_scale = np.asarray(
                [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]]
            )
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = point_level[idx_level]
            if idx_level > 0:
                res_points = int(
                    np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum()
                    - len(im_pts)
                )
                num_points_level = res_points

            ## to make the output score map derive more keypoints
            score_map = score_maps["map_" + str(idx_level + 1)]

            im_scores = geo_tools.apply_nms(score_map, self.nms_size)
            im_pts_tmp = geo_tools.get_point_coordinates(
                im_scores, num_points=num_points_level
            )
            im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

        im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
        im_pts = im_pts[:num_points]

        return im_pts

    def get_save_feat_dir(self):
        return self.save_feat_dir


def load_REKD(resize=None):
    args = get_config()
    args.load_dir = "release_group36_f2_s2_t2.log/best_model.pt"
    args.resize = resize
    model = MultiScaleFeatureExtractor(args)

    print("Model paramter : {} is loaded.".format(args.load_dir))
    return model
