from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import DEFAULT_MAE_LIMIT, DEFAULT_SSIM_LIMIT, DEFAULT_TOLERANCE


@dataclass(frozen=True)
class ComparisonMetrics:
    exact_match: bool
    tolerance_match: bool
    different_values: int
    different_pixels: int
    mae: float
    rmse: float
    max_difference: float
    psnr_db: float
    global_ssim: float


def _global_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64) / 255.0
    cand = candidate.astype(np.float64) / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    channel_scores: list[float] = []

    for channel in range(ref.shape[-1]):
        x = ref[..., channel].reshape(-1)
        y = cand[..., channel].reshape(-1)
        mu_x = float(x.mean())
        mu_y = float(y.mean())
        var_x = float(x.var())
        var_y = float(y.var())
        covariance = float(((x - mu_x) * (y - mu_y)).mean())
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        channel_scores.append(numerator / denominator if denominator else 1.0)

    return float(np.mean(channel_scores))


def compare_batches(
    reference_images: list[np.ndarray],
    candidate_images: list[np.ndarray],
) -> ComparisonMetrics:
    if len(reference_images) != len(candidate_images):
        return ComparisonMetrics(False, False, -1, -1, math.inf, math.inf, math.inf, 0.0, 0.0)

    reference = np.stack(reference_images).astype(np.float32)
    candidate = np.stack(candidate_images).astype(np.float32)
    if reference.shape != candidate.shape:
        return ComparisonMetrics(False, False, -1, -1, math.inf, math.inf, math.inf, 0.0, 0.0)

    absolute_difference = np.abs(reference - candidate)
    squared_difference = np.square(reference - candidate)
    mae = float(absolute_difference.mean())
    rmse = float(np.sqrt(squared_difference.mean()))
    max_difference = float(absolute_difference.max(initial=0.0))
    different_values = int(np.count_nonzero(absolute_difference > 0.0))
    different_pixels = int(np.count_nonzero(np.any(absolute_difference > 0.0, axis=-1)))
    exact_match = different_values == 0

    if rmse == 0.0:
        psnr = math.inf
    else:
        psnr = 20.0 * math.log10(255.0 / rmse)

    ssim = _global_ssim(reference.astype(np.uint8), candidate.astype(np.uint8))
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
