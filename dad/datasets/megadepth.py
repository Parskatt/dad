import os
from PIL import Image
import h5py
import math
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from tqdm import tqdm

import dad
from dad.augs import (
    get_tuple_transform_ops,
    get_depth_tuple_transform_ops,
)
from torch.utils.data import ConcatDataset


class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        scene_name=None,
        min_overlap=0.0,
        max_overlap=1.0,
        image_size=640,
        normalize=True,
        shake_t=32,
        rot_360=False,
        max_num_pairs=100_000,
    ) -> None:
        self.data_root = data_root
        self.scene_name = (
            os.path.splitext(scene_name)[0] + f"_{min_overlap}_{max_overlap}"
        )
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(image_size, image_size),
            normalize=normalize,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(image_size, image_size), normalize=False
        )
        self.image_size = image_size
        self.shake_t = shake_t
        self.rot_360 = rot_360

    def load_im(self, im_B, crop=None):
        im = Image.open(im_B)
        return im

    def rot_360_deg(self, im, depth, K, angle):
        C, H, W = im.shape
        im = tvf.rotate(im, angle, expand=True)
        depth = tvf.rotate(depth, angle, expand=True)
        radians = angle * math.pi / 180
        rot_mat = torch.tensor(
            [
                [math.cos(radians), math.sin(radians), 0],
                [-math.sin(radians), math.cos(radians), 0],
                [0, 0, 1.0],
            ]
        ).to(K.device)
        t_mat = torch.tensor([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1.0]]).to(K.device)
        neg_t_mat = torch.tensor([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1.0]]).to(
            K.device
        )
        transform = t_mat @ rot_mat @ neg_t_mat
        K = transform @ K
        return im, depth, K, transform

    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.image_size / wi, self.image_size / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=(2))
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        try:
            # read intrinsics of original size
            idx1, idx2 = self.pairs[pair_idx]
            K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(
                3, 3
            )
            K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(
                3, 3
            )

            # read and compute relative poses
            T1 = self.poses[idx1]
            T2 = self.poses[idx2]
            T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
                :4, :4
            ]  # (4, 4)

            # Load positive pair data
            im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
            depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
            im_A_ref = os.path.join(self.data_root, im_A)
            im_B_ref = os.path.join(self.data_root, im_B)
            depth_A_ref = os.path.join(self.data_root, depth1)
            depth_B_ref = os.path.join(self.data_root, depth2)
            im_A: Image.Image = self.load_im(im_A_ref)
            im_B: Image.Image = self.load_im(im_B_ref)
            depth_A = self.load_depth(depth_A_ref)
            depth_B = self.load_depth(depth_B_ref)

            # Recompute camera intrinsic matrix due to the resize
            W_A, H_A = im_A.width, im_A.height
            W_B, H_B = im_B.width, im_B.height

            K1 = self.scale_intrinsic(K1, W_A, H_A)
            K2 = self.scale_intrinsic(K2, W_B, H_B)

            # Process images
            im_A, im_B = self.im_transform_ops((im_A, im_B))
            depth_A, depth_B = self.depth_transform_ops(
                (depth_A[None, None], depth_B[None, None])
            )
            [im_A, depth_A], t_A = self.rand_shake(im_A, depth_A)
            [im_B, depth_B], t_B = self.rand_shake(im_B, depth_B)

            K1[:2, 2] += t_A
            K2[:2, 2] += t_B

            if self.rot_360:
                angle_A = np.random.choice([-90, 0, 90, 180])
                angle_B = np.random.choice([-90, 0, 90, 180])
                angle_A, angle_B = int(angle_A), int(angle_B)
                im_A, depth_A, K1, _ = self.rot_360_deg(
                    im_A, depth_A, K1, angle=angle_A
                )
                im_B, depth_B, K2, _ = self.rot_360_deg(
                    im_B, depth_B, K2, angle=angle_B
                )
            else:
                angle_A = 0
                angle_B = 0
            data_dict = {
                "im_A": im_A,
                "im_A_identifier": self.image_paths[idx1]
                .split("/")[-1]
                .split(".jpg")[0],
                "im_B": im_B,
                "im_B_identifier": self.image_paths[idx2]
                .split("/")[-1]
                .split(".jpg")[0],
                "im_A_depth": depth_A[0, 0],
                "im_B_depth": depth_B[0, 0],
                "pose_A": T1,
                "pose_B": T2,
                "K1": K1,
                "K2": K2,
                "T_1to2": T_1to2,
                "im_A_path": im_A_ref,
                "im_B_path": im_B_ref,
                "angle_A": angle_A,
                "angle_B": angle_B,
            }
        except Exception as e:
            dad.logger.warning(e)
            dad.logger.warning(f"Failed to load image pair {self.pairs[pair_idx]}")
            dad.logger.warning("Loading a random pair in scene instead")
            rand_ind = np.random.choice(range(len(self)))
            return self[rand_ind]
        return data_dict


class MegadepthBuilder:
    def __init__(self, data_root, loftr_ignore=True, imc21_ignore=True) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(
            [
                "0121.npy",
                "0133.npy",
                "0168.npy",
                "0178.npy",
                "0229.npy",
                "0349.npy",
                "0412.npy",
                "0430.npy",
                "0443.npy",
                "1001.npy",
                "5014.npy",
                "5015.npy",
                "5016.npy",
            ]
        )
        self.imc21_scenes = set(
            [
                "0008.npy",
                "0019.npy",
                "0021.npy",
                "0024.npy",
                "0025.npy",
                "0032.npy",
                "0063.npy",
                "1589.npy",
            ]
        )
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, split, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "all_scenes":
            scene_names = self.all_scenes
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in tqdm(scene_names):
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()

            scenes.append(
                MegadepthScene(
                    self.data_root,
                    scene_info,
                    scene_name=scene_name,
                    **kwargs,
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws

    def dedode_train_split(self, **kwargs):
        megadepth_train1 = self.build_scenes(
            split="train_loftr", min_overlap=0.01, **kwargs
        )
        megadepth_train2 = self.build_scenes(
            split="train_loftr", min_overlap=0.35, **kwargs
        )

        megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
        return megadepth_train

    def hard_train_split(self, **kwargs):
        megadepth_train = self.build_scenes(
            split="train_loftr", min_overlap=0.01, **kwargs
        )
        megadepth_train = ConcatDataset(megadepth_train)
        return megadepth_train

    def easy_train_split(self, **kwargs):
        megadepth_train = self.build_scenes(
            split="train_loftr", min_overlap=0.35, **kwargs
        )
        megadepth_train = ConcatDataset(megadepth_train)
        return megadepth_train

    def dedode_test_split(self, **kwargs):
        megadepth_test = self.build_scenes(
            split="test_loftr",
            min_overlap=0.01,
            **kwargs,
        )
        megadepth_test = ConcatDataset(megadepth_test)
        return megadepth_test
