from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import albumentations as A
import cv2
import numpy as np

from .config import AugmentationProfile
from .profiles import SampleParameters, parameters_for_sample


@dataclass(frozen=True)
class PreparedAlbumentations:
    transforms: tuple[A.Compose, ...]


def _fixed_affine(parameters: SampleParameters) -> Callable[..., np.ndarray]:
    def apply(image: np.ndarray, **_: object) -> np.ndarray:
        height, width = image.shape[:2]
        center = ((width - 1) * 0.5, (height - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, parameters.angle_degrees, parameters.scale)
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


def _build_transform(parameters: SampleParameters) -> A.Compose:
    transforms: list[A.BasicTransform] = []

    if parameters.horizontal_flip:
        transforms.append(A.HorizontalFlip(p=1.0))
    if parameters.vertical_flip:
        transforms.append(A.VerticalFlip(p=1.0))

    has_affine = (
        abs(parameters.angle_degrees) > 1e-12
        or abs(parameters.translate_x_fraction) > 1e-12
        or abs(parameters.translate_y_fraction) > 1e-12
        or abs(parameters.scale - 1.0) > 1e-12
    )
    if has_affine:
        transforms.append(A.Lambda(image=_fixed_affine(parameters), p=1.0))

    if abs(parameters.brightness_delta) > 1e-12 or abs(parameters.contrast_factor - 1.0) > 1e-12:
        contrast_delta = parameters.contrast_factor - 1.0
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(parameters.brightness_delta, parameters.brightness_delta),
                contrast_limit=(contrast_delta, contrast_delta),
                brightness_by_max=True,
                p=1.0,
            )
        )

    if parameters.blur_kernel > 1:
        transforms.append(
            A.GaussianBlur(
                blur_limit=(parameters.blur_kernel, parameters.blur_kernel),
                sigma_limit=(parameters.blur_sigma, parameters.blur_sigma),
                p=1.0,
            )
        )

    if not transforms:
        transforms.append(A.NoOp(p=1.0))

    return A.Compose(transforms, strict=True)


def prepare_albumentations(profile: AugmentationProfile, batch_size: int) -> PreparedAlbumentations:
    return PreparedAlbumentations(
        transforms=tuple(
            _build_transform(parameters_for_sample(profile, sample_index))
            for sample_index in range(batch_size)
        )
    )


def run_sequential(
    images: list[np.ndarray], prepared: PreparedAlbumentations
) -> list[np.ndarray]:
    return [
        transform(image=image)["image"]
        for image, transform in zip(images, prepared.transforms, strict=True)
    ]


def run_threaded(
    images: list[np.ndarray],
    prepared: PreparedAlbumentations,
    executor: ThreadPoolExecutor,
) -> list[np.ndarray]:
    def apply_one(item: tuple[np.ndarray, A.Compose]) -> np.ndarray:
        image, transform = item
        return transform(image=image)["image"]

    return list(executor.map(apply_one, zip(images, prepared.transforms, strict=True)))
