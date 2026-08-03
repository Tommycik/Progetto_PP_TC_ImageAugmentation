from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Callable

import numpy as np
from torch.utils import benchmark as torch_benchmark

from .config import (
    DEFAULT_MAE_LIMIT,
    DEFAULT_SSIM_LIMIT,
    DEFAULT_TOLERANCE,
)


# benchmark result containers
@dataclass(frozen=True)
class TimingStatistics:
    # raw timing samples and their summary values
    output: Any
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    coefficient_variation_percent: float


@dataclass(frozen=True)
class ComparisonMetrics:
    # exact and numerical comparison between two image batches
    exact_match: bool
    tolerance_match: bool
    different_values: int
    different_pixels: int
    mae: float
    rmse: float
    max_difference: float
    psnr_db: float
    global_ssim: float


# timing and correctness

def measure(function: Callable[[], Any], warmups: int, repetitions: int, synchronize: Callable[[], None] | None = None,) -> TimingStatistics:
    # execute warm-ups before starting the measured repetitions
    if repetitions < 1:
        raise ValueError("At least one measured repetition is required.")

    # keep the last output so timing and correctness use the same execution
    output: Any = None
    for _ in range(warmups):
        output = function()
        if synchronize is not None:
            synchronize()

    # perf_counter_ns provides a monotonic high-resolution host timer
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        output = function()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

    # calculate the same statistical values stored in the CSV
    mean_ms = statistics.fmean(samples)
    stddev_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    coefficient = (stddev_ms / mean_ms * 100.0) if mean_ms > 0.0 else 0.0
    return TimingStatistics(
        output=output,
        samples_ms=tuple(samples),
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stddev_ms=stddev_ms,
        coefficient_variation_percent=coefficient,
    )


def measure_torch_cpu(function: Callable[[], Any], threads: int, warmups: int, repetitions: int,) -> TimingStatistics:
    # PyTorch Timer applies the requested intra-operation thread-pool size
    # only while the Kornia CPU operation is measured.
    if threads < 1:
        raise ValueError("PyTorch CPU thread count must be positive.")
    if repetitions < 1:
        raise ValueError("At least one measured repetition is required.")

    # The mutable holder keeps the output of the last timed execution so the
    # same measured result can also be used for correctness verification.
    output_holder: dict[str, Any] = {"value": None}

    def execute_and_store() -> None:
        output_holder["value"] = function()

    timer = torch_benchmark.Timer(
        stmt="execute_and_store()",
        globals={"execute_and_store": execute_and_store},
        num_threads=threads,
    )

    # Explicit warm-ups initialize lazy operators before the stored samples.
    for _ in range(warmups):
        timer.timeit(number=1)

    samples = [
        timer.timeit(number=1).mean * 1000.0
        for _ in range(repetitions)
    ]
    mean_ms = statistics.fmean(samples)
    stddev_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    coefficient = (stddev_ms / mean_ms * 100.0) if mean_ms > 0.0 else 0.0

    return TimingStatistics(
        output=output_holder["value"],
        samples_ms=tuple(samples),
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stddev_ms=stddev_ms,
        coefficient_variation_percent=coefficient,
    )


def _global_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    # calculate one global SSIM score for every RGB channel
    ref = reference.astype(np.float64) / 255.0
    cand = candidate.astype(np.float64) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []

    for channel in range(ref.shape[-1]):
        x = ref[..., channel].reshape(-1)
        y = cand[..., channel].reshape(-1)
        mu_x = float(x.mean())
        mu_y = float(y.mean())
        var_x = float(x.var())
        var_y = float(y.var())
        covariance = float(((x - mu_x) * (y - mu_y)).mean())
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
            var_x + var_y + c2
        )
        scores.append(numerator / denominator if denominator else 1.0)

    return float(np.mean(scores))


def compare_batches(reference_images: list[np.ndarray], candidate_images: list[np.ndarray],) -> ComparisonMetrics:
    # compare equal-shaped batches using exact and perceptual measurements
    if len(reference_images) != len(candidate_images):
        return ComparisonMetrics(
            False,
            False,
            -1,
            -1,
            math.inf,
            math.inf,
            math.inf,
            0.0,
            0.0,
        )

    reference = np.stack(reference_images).astype(np.float32)
    candidate = np.stack(candidate_images).astype(np.float32)
    if reference.shape != candidate.shape:
        return ComparisonMetrics(
            False,
            False,
            -1,
            -1,
            math.inf,
            math.inf,
            math.inf,
            0.0,
            0.0,
        )

    # calculate all pixel differences once and reuse them for every metric
    absolute_difference = np.abs(reference - candidate)
    squared_difference = np.square(reference - candidate)
    mae = float(absolute_difference.mean())
    rmse = float(np.sqrt(squared_difference.mean()))
    max_difference = float(absolute_difference.max(initial=0.0))
    different_values = int(np.count_nonzero(absolute_difference > 0.0))
    different_pixels = int(
        np.count_nonzero(np.any(absolute_difference > 0.0, axis=-1))
    )
    exact_match = different_values == 0
    psnr = math.inf if rmse == 0.0 else 20.0 * math.log10(255.0 / rmse)
    ssim = _global_ssim(
        reference.astype(np.uint8),
        candidate.astype(np.uint8),
    )
    # accept either a small maximum error or a low-MAE high-SSIM result
    tolerance_match = (
        max_difference <= DEFAULT_TOLERANCE
        or (mae <= DEFAULT_MAE_LIMIT and ssim >= DEFAULT_SSIM_LIMIT)
    )

    return ComparisonMetrics(
        exact_match=exact_match,
        tolerance_match=tolerance_match,
        different_values=different_values,
        different_pixels=different_pixels,
        mae=mae,
        rmse=rmse,
        max_difference=max_difference,
        psnr_db=psnr,
        global_ssim=ssim,
    )