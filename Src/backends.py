from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import albumentations as alb
import cv2
import kornia
import numpy as np
import torch

from .config import AugmentationProfile, SampleParameters, parameters_for_sample


# backend parameter containers
@dataclass(frozen=True)
class PreparedAlbumentations:
    # one prepared transformation for every image in the batch
    transforms: tuple[alb.Compose, ...]


@dataclass(frozen=True)
class KorniaParameters:
    # tensors used by the Kornia CPU and CUDA paths
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


# Albumentations CPU path

def _fixed_affine(parameters: SampleParameters) -> Callable[..., np.ndarray]:
    # create the fixed OpenCV affine callback for one image
    def apply(image: np.ndarray, **_: object) -> np.ndarray:
        # rotation and scaling are applied around the image centre
        height, width = image.shape[:2]
        center = ((width - 1) * 0.5, (height - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(
            center,
            parameters.angle_degrees,
            parameters.scale,
        )
        # add translation after the rotation matrix is generated
        matrix[0, 2] += parameters.translate_x_fraction * width
        matrix[1, 2] += parameters.translate_y_fraction * height
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    return apply


def _build_albumentations_transform(parameters: SampleParameters) -> alb.Compose:
    # build one fixed pipeline without runtime random choices
    transforms: list[alb.BasicTransform] = []

    # append only the operations enabled by the selected profile
    if parameters.horizontal_flip:
        transforms.append(alb.HorizontalFlip(p=1.0))
    if parameters.vertical_flip:
        transforms.append(alb.VerticalFlip(p=1.0))

    # skip the affine call when rotation, translation and scale are neutral
    has_affine = (
        abs(parameters.angle_degrees) > 1e-12
        or abs(parameters.translate_x_fraction) > 1e-12
        or abs(parameters.translate_y_fraction) > 1e-12
        or abs(parameters.scale - 1.0) > 1e-12
    )
    if has_affine:
        transforms.append(alb.Lambda(image=_fixed_affine(parameters), p=1.0))

    # brightness and contrast are combined in one Albumentations operation
    has_colour = (
        abs(parameters.brightness_delta) > 1e-12
        or abs(parameters.contrast_factor - 1.0) > 1e-12
    )
    if has_colour:
        contrast_delta = parameters.contrast_factor - 1.0
        transforms.append(
            alb.RandomBrightnessContrast(
                brightness_limit=(
                    parameters.brightness_delta,
                    parameters.brightness_delta,
                ),
                contrast_limit=(contrast_delta, contrast_delta),
                brightness_by_max=True,
                p=1.0,
            )
        )

    # kernel size one represents a disabled blur
    if parameters.blur_kernel > 1:
        transforms.append(
            alb.GaussianBlur(
                blur_limit=(parameters.blur_kernel, parameters.blur_kernel),
                sigma_limit=(parameters.blur_sigma, parameters.blur_sigma),
                p=1.0,
            )
        )

    # Identity still uses a valid no-operation pipeline
    if not transforms:
        transforms.append(alb.NoOp(p=1.0))

    return alb.Compose(transforms, strict=True)


def prepare_albumentations(
    profile: AugmentationProfile,
    batch_size: int,
) -> PreparedAlbumentations:
    # prepare the complete batch before the timed repetitions
    return PreparedAlbumentations(
        transforms=tuple(
            _build_albumentations_transform(
                parameters_for_sample(profile, sample_index)
            )
            for sample_index in range(batch_size)
        )
    )


def run_albumentations_sequential(
    images: list[np.ndarray],
    prepared: PreparedAlbumentations,
) -> list[np.ndarray]:
    # process the batch in the original image order
    return [
        transform(image=image)["image"]
        for image, transform in zip(images, prepared.transforms, strict=True)
    ]


def run_albumentations_threaded(
    images: list[np.ndarray],
    prepared: PreparedAlbumentations,
    executor: ThreadPoolExecutor,
) -> list[np.ndarray]:
    # distribute independent images through the reused thread pool
    # one worker receives one image and its fixed transformation
    def apply_one(item: tuple[np.ndarray, alb.Compose]) -> np.ndarray:
        image, transform = item
        return transform(image=image)["image"]

    return list(
        executor.map(
            apply_one,
            zip(images, prepared.transforms, strict=True),
        )
    )


# NumPy and tensor conversion
def numpy_batch_to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    # stack RGB uint8 images and convert NHWC to BCHW float32
    batch = np.stack(images, axis=0)
    tensor = (
        torch.from_numpy(batch)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(torch.float32)
    )
    return tensor / 255.0


def tensor_to_numpy_batch(tensor: torch.Tensor) -> list[np.ndarray]:
    # clamp, round and convert the BCHW result back to RGB uint8 images
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



# Kornia CPU and CUDA path
def prepare_kornia_parameters(
    profile: AugmentationProfile,
    batch_size: int,
    width: int,
    height: int,
    device: torch.device,
) -> KorniaParameters:
    # create one parameter tensor for every transformation component
    # generate the same per-sample values used by Albumentations
    values = [parameters_for_sample(profile, index) for index in range(batch_size)]

    # small helper used to place all parameter arrays on the target device
    def tensor(
        data: list[float] | list[bool],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=device)

    return KorniaParameters(
        horizontal_flip=tensor(
            [item.horizontal_flip for item in values],
            torch.bool,
        ),
        vertical_flip=tensor(
            [item.vertical_flip for item in values],
            torch.bool,
        ),
        angles=tensor([item.angle_degrees for item in values]),
        translations_x=tensor(
            [item.translate_x_fraction * width for item in values]
        ),
        translations_y=tensor(
            [item.translate_y_fraction * height for item in values]
        ),
        scales=tensor([item.scale for item in values]),
        brightness=tensor([item.brightness_delta for item in values]),
        contrast=tensor([item.contrast_factor for item in values]),
        blur_kernel=profile.blur_kernel,
        blur_sigma=profile.blur_sigma,
    )


def apply_kornia(batch: torch.Tensor, parameters: KorniaParameters,) -> torch.Tensor:
    # execute the same batch pipeline on the tensor device
    with torch.inference_mode():
        # every operation creates the next stage of the batch pipeline
        output = batch
        batch_size, _, height, width = output.shape

        # select flipped images with a per-sample boolean mask
        if bool(parameters.horizontal_flip.any().item()):
            flipped = torch.flip(output, dims=(3,))
            mask = parameters.horizontal_flip.view(batch_size, 1, 1, 1)
            output = torch.where(mask, flipped, output)

        if bool(parameters.vertical_flip.any().item()):
            flipped = torch.flip(output, dims=(2,))
            mask = parameters.vertical_flip.view(batch_size, 1, 1, 1)
            output = torch.where(mask, flipped, output)

        # avoid warp_affine when the complete batch has neutral parameters
        has_affine = bool(
            (parameters.angles.abs() > 1e-12).any().item()
            or (parameters.translations_x.abs() > 1e-12).any().item()
            or (parameters.translations_y.abs() > 1e-12).any().item()
            or ((parameters.scales - 1.0).abs() > 1e-12).any().item()
        )
        if has_affine:
            # Kornia expects one image centre and one affine matrix per sample
            center = torch.empty(
                (batch_size, 2),
                dtype=output.dtype,
                device=output.device,
            )
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

        # apply contrast and brightness only when the profile enables them
        if bool(((parameters.contrast - 1.0).abs() > 1e-12).any().item()):
            factor = parameters.contrast.to(output.dtype).view(batch_size, 1, 1, 1)
            output = kornia.enhance.adjust_contrast(output, factor)

        if bool((parameters.brightness.abs() > 1e-12).any().item()):
            factor = parameters.brightness.to(output.dtype).view(
                batch_size,
                1,
                1,
                1,
            )
            output = kornia.enhance.adjust_brightness(
                output,
                factor,
                clip_output=True,
            )

        # Gaussian blur is executed as a batch operation
        if parameters.blur_kernel > 1:
            output = kornia.filters.gaussian_blur2d(
                output,
                kernel_size=(parameters.blur_kernel, parameters.blur_kernel),
                sigma=(parameters.blur_sigma, parameters.blur_sigma),
                border_type="reflect",
            )

        # keep the final tensor inside the valid normalized image range
        return output.clamp(0.0, 1.0)
