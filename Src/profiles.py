from __future__ import annotations
from dataclasses import dataclass

from .config import AugmentationProfile


@dataclass(frozen=True)
class SampleParameters:
    horizontal_flip: bool
    vertical_flip: bool
    angle_degrees: float
    translate_x_fraction: float
    translate_y_fraction: float
    scale: float
    brightness_delta: float
    contrast_factor: float
    blur_kernel: int
    blur_sigma: float


def parameters_for_sample(profile: AugmentationProfile, sample_index: int) -> SampleParameters:
    # Create deterministic parameters for a given batch index.
    direction = -1.0 if sample_index % 2 else 1.0
    secondary_direction = -1.0 if (sample_index // 2) % 2 else 1.0
    magnitude_cycle = (1.0, 0.65, 0.35, 0.85)
    magnitude = magnitude_cycle[sample_index % len(magnitude_cycle)]

    horizontal_flip = profile.horizontal_flip and sample_index % 2 == 0
    vertical_flip = profile.vertical_flip and sample_index % 3 == 0
    angle = profile.angle_degrees * direction * magnitude
    translate_x = profile.translate_fraction * direction * magnitude
    translate_y = profile.translate_fraction * secondary_direction * magnitude
    scale = 1.0 + profile.scale_delta * secondary_direction * magnitude
    contrast = 1.0 + profile.contrast_delta * secondary_direction * magnitude
    brightness = profile.brightness_delta * direction * magnitude

    return SampleParameters(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        angle_degrees=angle,
        translate_x_fraction=translate_x,
        translate_y_fraction=translate_y,
        scale=scale,
        brightness_delta=brightness,
        contrast_factor=contrast,
        blur_kernel=profile.blur_kernel,
        blur_sigma=profile.blur_sigma,
    )
