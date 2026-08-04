from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


# directories used by the menu, benchmark and preview generator
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "Output"
RESULTS_ROOT: Final[Path] = OUTPUT_ROOT / "Results"
PREVIEWS_ROOT: Final[Path] = OUTPUT_ROOT / "Previews"


# limits used for cross-library output verification
DEFAULT_TOLERANCE: Final[int] = 3
DEFAULT_MAE_LIMIT: Final[float] = 2.0
DEFAULT_SSIM_LIMIT: Final[float] = 0.990

# configuration containers

@dataclass(frozen=True)
class AugmentationProfile:
    # maximum transformation values shared by both libraries
    name: str
    description: str
    horizontal_flip: bool = False
    vertical_flip: bool = False
    angle_degrees: float = 0.0
    translate_fraction: float = 0.0
    scale_delta: float = 0.0
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    blur_kernel: int = 1
    blur_sigma: float = 0.0


@dataclass(frozen=True)
class SampleParameters:
    # concrete transformation values applied to one image
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


@dataclass(frozen=True)
class BenchmarkPlan:
    # complete benchmark matrix
    name: str
    profiles: tuple[str, ...]
    workloads: tuple[tuple[int, tuple[int, ...]], ...]
    thread_counts: tuple[int, ...]
    repetitions: int
    warmups: int

# augmentation profiles
# each profile uses operations implemented by Albumentations and Kornia
PROFILES: Final[dict[str, AugmentationProfile]] = {
    "Identity": AugmentationProfile(
        name="Identity",
        description="No augmentation. Measures framework and batching overhead.",
    ),
    "GeometricLight": AugmentationProfile(
        name="GeometricLight",
        description="Horizontal flip with mild rotation, translation and rescaling.",
        horizontal_flip=True,
        angle_degrees=8.0,
        translate_fraction=0.05,
        scale_delta=0.05,
    ),
    "GeometricStrong": AugmentationProfile(
        name="GeometricStrong",
        description="Horizontal and vertical flips with stronger affine motion.",
        horizontal_flip=True,
        vertical_flip=True,
        angle_degrees=18.0,
        translate_fraction=0.12,
        scale_delta=0.12,
    ),
    "Color": AugmentationProfile(
        name="Color",
        description="Brightness and contrast variation.",
        brightness_delta=0.16,
        contrast_delta=0.22,
    ),
    "Blur": AugmentationProfile(
        name="Blur",
        description="Fixed Gaussian blur with a medium kernel.",
        blur_kernel=7,
        blur_sigma=1.6,
    ),
    "MixedLight": AugmentationProfile(
        name="MixedLight",
        description="Light geometric motion, colour changes and mild blur.",
        horizontal_flip=True,
        angle_degrees=10.0,
        translate_fraction=0.06,
        scale_delta=0.06,
        brightness_delta=0.10,
        contrast_delta=0.14,
        blur_kernel=5,
        blur_sigma=1.0,
    ),
    "MixedStrong": AugmentationProfile(
        name="MixedStrong",
        description="Strong geometric motion, colour changes and stronger blur.",
        horizontal_flip=True,
        vertical_flip=True,
        angle_degrees=18.0,
        translate_fraction=0.12,
        scale_delta=0.12,
        brightness_delta=0.22,
        contrast_delta=0.30,
        blur_kernel=9,
        blur_sigma=2.0,
    ),
}

# benchmark plan

# one complete run evaluates every profile, resolution, batch and CPU thread count
FULL_BENCHMARK_PLAN: Final[BenchmarkPlan] = BenchmarkPlan(
    name="Full",
    profiles=tuple(PROFILES.keys()),
    workloads=(
        (512, (1, 4, 8, 16, 32, 64)),
        (1024, (1, 4, 8, 16, 32)),
        (2048, (1, 4, 8)),
        (4096, (1, 2)),
    ),
    # used by Albumentations workers and Kornia/PyTorch intra-op threads
    thread_counts=(1, 2, 4, 6, 8, 12),
    repetitions=5,
    warmups=2,
)


# deterministic parameters

# create a reproducible set of parameters for each sample in a profile
def parameters_for_sample(
    profile: AugmentationProfile,
    sample_index: int,
) -> SampleParameters:
    # alternate direction and magnitude using only the sample index
    direction = -1.0 if sample_index % 2 else 1.0
    secondary_direction = -1.0 if (sample_index // 2) % 2 else 1.0
    # the cycle gives different strengths while remaining reproducible
    magnitude_cycle = (1.0, 0.65, 0.35, 0.85)
    magnitude = magnitude_cycle[sample_index % len(magnitude_cycle)]

    # convert the profile maxima into the values used by this sample
    return SampleParameters(
        horizontal_flip=profile.horizontal_flip and sample_index % 2 == 0,
        vertical_flip=profile.vertical_flip and sample_index % 3 == 0,
        angle_degrees=profile.angle_degrees * direction * magnitude,
        translate_x_fraction=profile.translate_fraction * direction * magnitude,
        translate_y_fraction=(
            profile.translate_fraction * secondary_direction * magnitude
        ),
        scale=1.0 + profile.scale_delta * secondary_direction * magnitude,
        brightness_delta=profile.brightness_delta * direction * magnitude,
        contrast_factor=(
            1.0 + profile.contrast_delta * secondary_direction * magnitude
        ),
        blur_kernel=profile.blur_kernel,
        blur_sigma=profile.blur_sigma,
    )


def ensure_output_directories() -> None:
    # create the output folders before the menu starts
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
