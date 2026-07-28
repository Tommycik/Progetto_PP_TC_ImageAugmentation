from __future__ import annotations

from dataclasses import dataclass

import kornia
import torch

from .config import AugmentationProfile
from .profiles import parameters_for_sample


@dataclass(frozen=True)
class KorniaParameters:
    horizontal_flip: torch.Tensor
    vertical_flip: torch.Tensor
    angles: torch.Tensor
    translations_x: torch.Tensor
    translations_y: torch.Tensor
    scales: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    blur_kernel: int
    blur_sigma: float


def numpy_batch_to_tensor(images: list[object]) -> torch.Tensor:
    import numpy as np

    batch = np.stack(images, axis=0)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(torch.float32)
    return tensor / 255.0


def tensor_to_numpy_batch(tensor: torch.Tensor) -> list[object]:
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [item.copy() for item in array]


def prepare_kornia_parameters(
    profile: AugmentationProfile,
    batch_size: int,
    width: int,
    height: int,
    device: torch.device,
) -> KorniaParameters:
    values = [parameters_for_sample(profile, index) for index in range(batch_size)]

    def tensor(data: list[float] | list[bool], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=device)

    return KorniaParameters(
        horizontal_flip=tensor([item.horizontal_flip for item in values], torch.bool),
        vertical_flip=tensor([item.vertical_flip for item in values], torch.bool),
        angles=tensor([item.angle_degrees for item in values]),
        translations_x=tensor([item.translate_x_fraction * width for item in values]),
        translations_y=tensor([item.translate_y_fraction * height for item in values]),
        scales=tensor([item.scale for item in values]),
        brightness=tensor([item.brightness_delta for item in values]),
        contrast=tensor([item.contrast_factor for item in values]),
        blur_kernel=profile.blur_kernel,
        blur_sigma=profile.blur_sigma,
    )


def apply_kornia(batch: torch.Tensor, parameters: KorniaParameters) -> torch.Tensor:
    with torch.inference_mode():
        output = batch
        batch_size, _, height, width = output.shape

        if bool(parameters.horizontal_flip.any()):
            flipped = torch.flip(output, dims=(3,))
            mask = parameters.horizontal_flip.view(batch_size, 1, 1, 1)
            output = torch.where(mask, flipped, output)

        if bool(parameters.vertical_flip.any()):
            flipped = torch.flip(output, dims=(2,))
            mask = parameters.vertical_flip.view(batch_size, 1, 1, 1)
            output = torch.where(mask, flipped, output)

        has_affine = bool(
            (parameters.angles.abs() > 1e-12).any()
            or (parameters.translations_x.abs() > 1e-12).any()
            or (parameters.translations_y.abs() > 1e-12).any()
            or ((parameters.scales - 1.0).abs() > 1e-12).any()
        )
        if has_affine:
            center = torch.empty((batch_size, 2), dtype=output.dtype, device=output.device)
            center[:, 0] = (width - 1) * 0.5
            center[:, 1] = (height - 1) * 0.5
            scale_xy = parameters.scales.to(output.dtype).view(-1, 1).repeat(1, 2)
            matrix = kornia.geometry.transform.get_rotation_matrix2d(
                center,
                parameters.angles.to(output.dtype),
                scale_xy,
            )
            matrix[:, 0, 2] += parameters.translations_x.to(output.dtype)
            matrix[:, 1, 2] += parameters.translations_y.to(output.dtype)
            output = kornia.geometry.transform.warp_affine(
                output,
                matrix,
                dsize=(height, width),
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )

        if bool(((parameters.contrast - 1.0).abs() > 1e-12).any()):
            factor = parameters.contrast.to(output.dtype).view(batch_size, 1, 1, 1)
            output = kornia.enhance.adjust_contrast(output, factor)

        if bool((parameters.brightness.abs() > 1e-12).any()):
            factor = parameters.brightness.to(output.dtype).view(batch_size, 1, 1, 1)
            output = kornia.enhance.adjust_brightness(output, factor, clip_output=True)

        if parameters.blur_kernel > 1:
            output = kornia.filters.gaussian_blur2d(
                output,
                kernel_size=(parameters.blur_kernel, parameters.blur_kernel),
                sigma=(parameters.blur_sigma, parameters.blur_sigma),
                border_type="reflect",
            )

        return output.clamp(0.0, 1.0)
