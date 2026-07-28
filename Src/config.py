from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "Output"
RESULTS_ROOT: Final[Path] = OUTPUT_ROOT / "Results"
PREVIEWS_ROOT: Final[Path] = OUTPUT_ROOT / "Previews"

DEFAULT_SEED: Final[int] = 137
DEFAULT_TOLERANCE: Final[int] = 3
DEFAULT_MAE_LIMIT: Final[float] = 2.0
DEFAULT_SSIM_LIMIT: Final[float] = 0.990


@dataclass(frozen=True)
class AugmentationProfile:
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


PROFILES: Final[dict[str, AugmentationProfile]] = {
    "Identity": AugmentationProfile(
        name="Identity",
        description="No augmentation. Measures framework and batching overhead.",
    ),
    "Geometric": AugmentationProfile(
        name="Geometric",
        description="Flips, rotation, translation and a small scale change.",
        horizontal_flip=True,
        vertical_flip=True,
        angle_degrees=12.0,
        translate_fraction=0.08,
        scale_delta=0.08,
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
    "MixedStrong": AugmentationProfile(
        name="MixedStrong",
        description="Geometric, colour and blur transformations combined.",
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


@dataclass(frozen=True)
class BenchmarkPlan:
    name: str
    profiles: tuple[str, ...]
    workloads: tuple[tuple[int, tuple[int, ...]], ...]
    thread_counts: tuple[int, ...]
    gpu_blocks: tuple[tuple[int, int], ...]
    repetitions: int
    warmups: int


QUICK_PLAN: Final[BenchmarkPlan] = BenchmarkPlan(
    name="Quick",
    profiles=("Identity", "MixedStrong"),
    workloads=(
        (512, (1, 4, 8)),
        (1024, (1, 4, 8)),
    ),
    thread_counts=(1, 4, 8, 12),
    gpu_blocks=((8, 8), (16, 16), (32, 8), (32, 16)),
    repetitions=3,
    warmups=1,
)

FULL_PLAN: Final[BenchmarkPlan] = BenchmarkPlan(
    name="Full",
    profiles=tuple(PROFILES.keys()),
    workloads=(
        (512, (1, 4, 8, 16, 32)),
        (1024, (1, 4, 8, 16)),
        (2048, (1, 4)),
    ),
    thread_counts=(1, 2, 4, 6, 8, 12),
    gpu_blocks=(
        (8, 8),
        (16, 16),
        (32, 8),
        (8, 32),
        (32, 16),
        (16, 32),
        (32, 32),
        (256, 1),
        (1, 256),
    ),
    repetitions=5,
    warmups=2,
)


def ensure_output_directories() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
