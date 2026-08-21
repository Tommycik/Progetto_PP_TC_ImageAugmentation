from __future__ import annotations

import csv
from datetime import datetime
import math
from typing import Any

import torch

from .config import BenchmarkPlan, PROFILES
from .metrics import ComparisonMetrics, TimingStatistics


# CSV schema
# every backend writes the same timing, speedup and correctness fields
CSV_FIELDS = [
    "Timestamp",
    "Plan",
    "InputSource",
    "Profile",
    "ProfileDescription",
    "Resolution",
    "BatchSize",
    "Backend",
    "Library",
    "Device",
    "Workers",
    "Parameter",
    "TimerScope",
    "Warmups",
    "Repetitions",
    "TimeMean_ms",
    "TimeMedian_ms",
    "TimeMin_ms",
    "TimeMax_ms",
    "StdDev_ms",
    "CoefficientVariation_percent",
    "SequentialBaseline_ms",
    "BestCpu_ms",
    "Speedup_vs_Sequential",
    "Speedup_vs_BestCPU",
    "ParallelBaseline_ms",
    "ParallelSpeedup",
    "ParallelEfficiency",
    "Throughput_images_s",
    "Throughput_MPixels_s",
    "ReferenceBackend",
    "ComparisonKind",
    "ExactMatch",
    "ToleranceMatch",
    "DifferentValues",
    "DifferentPixels",
    "MAE",
    "RMSE",
    "MaxDifference",
    "PSNR_dB",
    "GlobalSSIM",
    "CudaAvailable",
    "GpuName",
    "CudaPeakMemory_MB",
    "HorizontalFlip",
    "VerticalFlip",
    "AngleDegrees",
    "TranslateFraction",
    "ScaleDelta",
    "BrightnessDelta",
    "ContrastDelta",
    "BlurKernel",
    "BlurSigma",
    "Status",
    "Notes",
]

# CSV row helpers
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _statistics_values(statistics_data: TimingStatistics | None) -> dict[str, str]:
    if statistics_data is None:
        return {
            "TimeMean_ms": "",
            "TimeMedian_ms": "",
            "TimeMin_ms": "",
            "TimeMax_ms": "",
            "StdDev_ms": "",
            "CoefficientVariation_percent": "",
        }
    return {
        "TimeMean_ms": _format_number(statistics_data.mean_ms),
        "TimeMedian_ms": _format_number(statistics_data.median_ms),
        "TimeMin_ms": _format_number(statistics_data.min_ms),
        "TimeMax_ms": _format_number(statistics_data.max_ms),
        "StdDev_ms": _format_number(statistics_data.stddev_ms),
        "CoefficientVariation_percent": _format_number(
            statistics_data.coefficient_variation_percent
        ),
    }


def _metric_values(metrics: ComparisonMetrics | None) -> dict[str, str]:
    if metrics is None:
        return {
            "ExactMatch": "",
            "ToleranceMatch": "",
            "DifferentValues": "",
            "DifferentPixels": "",
            "MAE": "",
            "RMSE": "",
            "MaxDifference": "",
            "PSNR_dB": "",
            "GlobalSSIM": "",
        }
    return {
        "ExactMatch": "YES" if metrics.exact_match else "NO",
        "ToleranceMatch": "YES" if metrics.tolerance_match else "NO",
        "DifferentValues": str(metrics.different_values),
        "DifferentPixels": str(metrics.different_pixels),
        "MAE": _format_number(metrics.mae),
        "RMSE": _format_number(metrics.rmse),
        "MaxDifference": _format_number(metrics.max_difference),
        "PSNR_dB": _format_number(metrics.psnr_db),
        "GlobalSSIM": _format_number(metrics.global_ssim),
    }


def _base_row(plan: BenchmarkPlan, input_source: str, profile_name: str, resolution: int, batch_size: int,) -> dict[str, str]:
    # fields shared by every backend in the current workload
    profile = PROFILES[profile_name]
    return {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Plan": plan.name,
        "InputSource": input_source,
        "Profile": profile.name,
        "ProfileDescription": profile.description,
        "Resolution": f"{resolution}x{resolution}",
        "BatchSize": str(batch_size),
        "CudaAvailable": "YES" if torch.cuda.is_available() else "NO",
        "GpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "HorizontalFlip": "YES" if profile.horizontal_flip else "NO",
        "VerticalFlip": "YES" if profile.vertical_flip else "NO",
        "AngleDegrees": _format_number(profile.angle_degrees),
        "TranslateFraction": _format_number(profile.translate_fraction),
        "ScaleDelta": _format_number(profile.scale_delta),
        "BrightnessDelta": _format_number(profile.brightness_delta),
        "ContrastDelta": _format_number(profile.contrast_delta),
        "BlurKernel": str(profile.blur_kernel),
        "BlurSigma": _format_number(profile.blur_sigma),
    }


def _result_row(
    *,
    base: dict[str, str],
    backend: str,
    library: str,
    device: str,
    workers: int | None,
    parameter: str,
    timer_scope: str,
    warmups: int,
    repetitions: int,
    statistics_data: TimingStatistics | None,
    sequential_ms: float,
    best_cpu_ms: float,
    parallel_baseline_ms: float | None,
    reference_backend: str,
    comparison_kind: str,
    metrics: ComparisonMetrics | None,
    peak_memory_mb: float | None = None,
    status: str = "SUCCESS",
    notes: str = "",
) -> dict[str, str]:
    # start from an empty row so skipped backends keep the same schema
    row = {field: "" for field in CSV_FIELDS}
    row.update(base)
    row.update(
        {
            "Backend": backend,
            "Library": library,
            "Device": device,
            "Workers": "" if workers is None else str(workers),
            "Parameter": parameter,
            "TimerScope": timer_scope,
            "Warmups": str(warmups),
            "Repetitions": str(repetitions),
            "SequentialBaseline_ms": _format_number(sequential_ms),
            "BestCpu_ms": _format_number(best_cpu_ms),
            "ParallelBaseline_ms": _format_number(parallel_baseline_ms),
            "ReferenceBackend": reference_backend,
            "ComparisonKind": comparison_kind,
            "CudaPeakMemory_MB": _format_number(peak_memory_mb),
            "Status": status,
            "Notes": notes,
        }
    )
    row.update(_statistics_values(statistics_data))
    row.update(_metric_values(metrics))

    # derived speedup and throughput values require a valid measured mean
    if statistics_data is not None and statistics_data.mean_ms > 0.0:
        mean_ms = statistics_data.mean_ms
        batch_size = int(base["BatchSize"])
        resolution = int(base["Resolution"].split("x", maxsplit=1)[0])
        row["Speedup_vs_Sequential"] = _format_number(sequential_ms / mean_ms)
        row["Speedup_vs_BestCPU"] = _format_number(best_cpu_ms / mean_ms)
        row["Throughput_images_s"] = _format_number(batch_size * 1000.0 / mean_ms)
        row["Throughput_MPixels_s"] = _format_number(
            batch_size * resolution * resolution / mean_ms / 1000.0
        )
        if parallel_baseline_ms is not None and parallel_baseline_ms > 0.0:
            parallel_speedup = parallel_baseline_ms / mean_ms
            row["ParallelSpeedup"] = _format_number(parallel_speedup)
            if workers is not None and workers > 0:
                row["ParallelEfficiency"] = _format_number(
                    parallel_speedup / workers
                )

    return row


def _write_row( writer: csv.DictWriter, file_handle: Any, row: dict[str, str],) -> None:
    # flush every row so completed workloads survive an interrupted run
    writer.writerow(row)
    file_handle.flush()