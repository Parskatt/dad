import random
import warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2


# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(
            TupleResize(resize, mode=InterpolationMode.BILINEAR, antialias=False)
        )
    return TupleCompose(ops)


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False, clahe=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, antialias=True))
    if clahe:
        ops.append(TupleClahe())
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)


class Clahe:
    def __init__(self, cliplimit=2, blocksize=8) -> None:
        self.clahe = cv2.createCLAHE(cliplimit, (blocksize, blocksize))

    def __call__(self, im):
        im_hsv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2HSV)
        im_v = self.clahe.apply(im_hsv[:, :, 2])
        im_hsv[..., 2] = im_v
        im_clahe = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(im_clahe)


class TupleClahe:
    def __init__(self, cliplimit=8, blocksize=8) -> None:
        self.clahe = Clahe(cliplimit, blocksize)

    def __call__(self, ims):
        return [self.clahe(im) for im in ims]


class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC, antialias=None):
        self.size = size
        self.resize = transforms.Resize(size, mode, antialias=antialias)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class Normalize:
    def __call__(self, im):
        mean = im.mean(dim=(1, 2), keepdims=True)
        std = im.std(dim=(1, 2), keepdims=True)
        return (im - mean) / std


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        c, h, w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels {c=} > 3, assuming first 3 are rgb")
        return [self.normalize(im[:3]) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def pad_kps(kps: torch.Tensor, pad_num_kps: int, value: int = -1):
    assert len(kps.shape) == 2
    N = len(kps)
    padded_kps = value * torch.ones((pad_num_kps, 2)).to(kps)
    padded_kps[:N] = kps
    return padded_kps


def crop(img: Image.Image, x: int, y: int, crop_size: int):
    width, height = img.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Image dimensions must be at least {crop_size}x{crop_size}")
    cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
    return cropped_img


def random_crop(img: Image.Image, crop_size: int):
    width, height = img.size

    if width < crop_size or height < crop_size:
        raise ValueError(f"Image dimensions must be at least {crop_size}x{crop_size}")

    max_x = width - crop_size
    max_y = height - crop_size

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    cropped_img = img.crop((x, y, x + crop_size, y + crop_size))
    return cropped_img, (x, y)


def luminance_negation(pil_img):
    # Convert PIL RGB to numpy array
    rgb_array = np.array(pil_img)

    # Convert RGB to BGR (OpenCV format)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Convert BGR to LAB
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Negate L channel
    lab[:, :, 0] = 255 - lab[:, :, 0]

    # Convert back to BGR
    bgr_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert BGR back to RGB
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)

    # Convert numpy array back to PIL Image
    return Image.fromarray(rgb_result)
