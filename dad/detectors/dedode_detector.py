import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as transforms
from PIL import Image
from dad.utils import get_best_device, sample_keypoints, check_not_i16

from dad.types import Detector


class DeDoDeDetector(Detector):
    def __init__(
        self,
        *args,
        encoder: nn.Module,
        decoder: nn.Module,
        resize: int,
        nms_size: int,
        subpixel: bool,
        subpixel_temp: float,
        keep_aspect_ratio: bool,
        remove_borders: bool,
        increase_coverage: bool,
        coverage_pow: float,
        coverage_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.remove_borders = remove_borders
        self.resize = resize
        self.increase_coverage = increase_coverage
        self.coverage_pow = coverage_pow
        self.coverage_size = coverage_size
        self.nms_size = nms_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.subpixel = subpixel
        self.subpixel_temp = subpixel_temp

    @property
    def topleft(self):
        return 0.5

    def forward_impl(
        self,
        images,
    ):
        features, sizes = self.encoder(images)
        logits = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            logits = (
                logits + delta_logits.float()
            )  # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(
                    logits, size=size, mode="bicubic", align_corners=False
                )
                context = F.interpolate(
                    context.float(), size=size, mode="bilinear", align_corners=False
                )
        return logits.float()

    def forward(self, batch) -> dict[str, torch.Tensor]:
        # wraps internal forward impl to handle
        # different types of batches etc.
        if "im_A" in batch:
            images = torch.cat((batch["im_A"], batch["im_B"]))
        else:
            images = batch["image"]
        scoremap = self.forward_impl(images)
        return {"scoremap": scoremap}

    @torch.inference_mode()
    def detect(
        self, batch, *, num_keypoints, return_dense_probs=False
    ) -> dict[str, torch.Tensor]:
        self.train(False)
        scoremap = self.forward(batch)["scoremap"]
        B, K, H, W = scoremap.shape
        dense_probs = (
            scoremap.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
        )
        dense_probs = dense_probs.reshape(B, H, W)
        keypoints, confidence = sample_keypoints(
            dense_probs,
            use_nms=True,
            nms_size=self.nms_size,
            sample_topk=True,
            num_samples=num_keypoints,
            return_probs=True,
            increase_coverage=self.increase_coverage,
            remove_borders=self.remove_borders,
            coverage_pow=self.coverage_pow,
            coverage_size=self.coverage_size,
            subpixel=self.subpixel,
            subpixel_temp=self.subpixel_temp,
            scoremap=scoremap.reshape(B, H, W),
        )
        result = {"keypoints": keypoints, "keypoint_probs": confidence}
        if return_dense_probs:
            result["dense_probs"] = dense_probs
        return result

    def load_image(self, im_path, device=get_best_device()) -> dict[str, torch.Tensor]:
        pil_im = Image.open(im_path)
        check_not_i16(pil_im)
        pil_im = pil_im.convert("RGB")
        if self.keep_aspect_ratio:
            W, H = pil_im.size
            scale = self.resize / max(W, H)
            W = int((scale * W) // 8 * 8)
            H = int((scale * H) // 8 * 8)
        else:
            H, W = self.resize, self.resize
        pil_im = pil_im.resize((W, H))
        standard_im = np.array(pil_im) / 255.0
        return {
            "image": self.normalizer(torch.from_numpy(standard_im).permute(2, 0, 1))
            .float()
            .to(device)[None]
        }


class Decoder(nn.Module):
    def __init__(
        self, layers, *args, super_resolution=False, num_prototypes=1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = self.layers.keys()
        self.super_resolution = super_resolution
        self.num_prototypes = num_prototypes

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.num_prototypes],
            stuff[:, self.num_prototypes :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=True,
        kernel_size=5,
        hidden_blocks=5,
        amp=True,
        residual=False,
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.residual = residual

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, (
                "outdim must be divisible by indim for depthwise"
            )
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        with torch.autocast(device_type=feats.device.type, enabled=self.amp, dtype=self.amp_dtype):
            x0 = self.block1(feats)
            x = self.hidden_blocks(x0)
            if self.residual:
                x = (x + x0) / 1.4
            x = self.out_conv(x)
            return x


class VGG19(nn.Module):
    def __init__(self, amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class VGG(nn.Module):
    def __init__(self, size="19", amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn().features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn().features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


def dedode_detector_S():
    residual = True
    hidden_blocks = 3
    amp_dtype = torch.float16
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG(size="11", amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


def dedode_detector_B():
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16
    amp = True
    NUM_PROTOTYPES = 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


def dedode_detector_L():
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = (
        torch.float16
    )  # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    return encoder, decoder


class DaD(DeDoDeDetector):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        resize=1024,
        nms_size=3,
        remove_borders=False,
        increase_coverage=False,
        coverage_pow=None,
        coverage_size=None,
        subpixel=True,
        subpixel_temp=0.5,
        keep_aspect_ratio=True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            encoder=encoder,
            decoder=decoder,
            resize=resize,
            nms_size=nms_size,
            remove_borders=remove_borders,
            increase_coverage=increase_coverage,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            keep_aspect_ratio=keep_aspect_ratio,
            subpixel_temp=subpixel_temp,
            **kwargs,
        )


class DeDoDev2(DeDoDeDetector):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        resize=784,
        nms_size=3,
        remove_borders=False,
        increase_coverage=True,
        coverage_pow=0.5,
        coverage_size=51,
        subpixel=False,
        subpixel_temp=None,
        keep_aspect_ratio=False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            encoder=encoder,
            decoder=decoder,
            resize=resize,
            nms_size=nms_size,
            remove_borders=remove_borders,
            increase_coverage=increase_coverage,
            coverage_pow=coverage_pow,
            coverage_size=coverage_size,
            subpixel=subpixel,
            keep_aspect_ratio=keep_aspect_ratio,
            subpixel_temp=subpixel_temp,
            **kwargs,
        )


def load_DaD(resize=1024, pretrained=True, weights_path=None) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth"
        )
    device = get_best_device()
    encoder, decoder = dedode_detector_S()
    model = DaD(encoder, decoder, resize=resize).to(device)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            weights_path, weights_only=False, map_location=device
        )
        model.load_state_dict(weights)
    return model


def load_DaDLight(resize=1024, weights_path=None) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad_light.pth"
        )
    return load_DaD(
        resize=resize,
        pretrained=True,
        weights_path=weights_path,
    )


def load_DaDDark(resize=1024, weights_path=None) -> DaD:
    if weights_path is None:
        weights_path = (
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad_dark.pth"
        )
    return load_DaD(
        resize=resize,
        pretrained=True,
        weights_path=weights_path,
    )


def load_dedode_v2() -> DeDoDev2:
    device = get_best_device()
    weights = torch.hub.load_state_dict_from_url(
        "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
        map_location=device,
    )

    encoder, decoder = dedode_detector_L()
    model = DeDoDev2(encoder, decoder).to(device)
    model.load_state_dict(weights)
    return model
