from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import cv2
import numpy as np
import torch

from .backends import (
    apply_kornia,
    numpy_batch_to_tensor,
    prepare_albumentations,
    prepare_kornia_parameters,
    run_albumentations_sequential,
    run_albumentations_threaded,
    tensor_to_numpy_batch,
)
from .config import (
    BenchmarkPlan,
    DEFAULT_MAE_LIMIT,
    DEFAULT_SSIM_LIMIT,
    DEFAULT_TOLERANCE,
    PROFILES,
    RESULTS_ROOT,
)


# -----------------------------------------------------------------------------
# CSV schema
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# benchmark result containers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# timing and correctness
# -----------------------------------------------------------------------------

def measure(
    function: Callable[[], Any],
    warmups: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> TimingStatistics:
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


def compare_batches(
    reference_images: list[np.ndarray],
    candidate_images: list[np.ndarray],
) -> ComparisonMetrics:
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


# -----------------------------------------------------------------------------
# CSV row helpers
# -----------------------------------------------------------------------------

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


def _base_row(
    plan: BenchmarkPlan,
    input_source: str,
    profile_name: str,
    resolution: int,
    batch_size: int,
) -> dict[str, str]:
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
        if workers is not None and workers > 0:
            row["ParallelEfficiency"] = _format_number(
                sequential_ms / mean_ms / workers
            )

    return row


def _write_row(
    writer: csv.DictWriter,
    file_handle: Any,
    row: dict[str, str],
) -> None:
    # flush every row so completed workloads survive an interrupted run
    writer.writerow(row)
    file_handle.flush()


# -----------------------------------------------------------------------------
# complete benchmark execution
# -----------------------------------------------------------------------------

def run_benchmark(
    source_images: list[np.ndarray],
    input_source: str,
    plan: BenchmarkPlan,
    build_batch: Callable[[list[np.ndarray], int, int], list[np.ndarray]],
) -> Path:
    # execute every profile, resolution, batch and backend configuration
    cv2.setNumThreads(1)
    result_path = (
        RESULTS_ROOT
        / f"augmentation_benchmark_{plan.name.lower()}_{_timestamp()}.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # devices are created once and reused by all workloads
    cpu_device = torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device("cuda:0") if cuda_available else None
    # one workload is a profile, resolution and batch-size combination
    total_workloads = (
        sum(len(batch_sizes) for _, batch_sizes in plan.workloads)
        * len(plan.profiles)
    )
    workload_index = 0

    with result_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for profile_name in plan.profiles:
            profile = PROFILES[profile_name]
            for resolution, batch_sizes in plan.workloads:
                for batch_size in batch_sizes:
                    workload_index += 1
                    print(
                        f"\n[{workload_index}/{total_workloads}] "
                        f"Profile={profile_name}, "
                        f"Resolution={resolution}x{resolution}, "
                        f"Batch={batch_size}"
                    )

                    # prepare the input batch before any timed backend call
                    images = build_batch(source_images, resolution, batch_size)
                    prepared = prepare_albumentations(profile, batch_size)
                    base = _base_row(
                        plan,
                        input_source,
                        profile_name,
                        resolution,
                        batch_size,
                    )

                    # sequential Albumentations is the CPU correctness reference
                    sequential_operation = partial(
                        run_albumentations_sequential,
                        images,
                        prepared,
                    )
                    sequential_stats = measure(
                        sequential_operation,
                        warmups=plan.warmups,
                        repetitions=plan.repetitions,
                    )
                    sequential_output = sequential_stats.output
                    sequential_ms = sequential_stats.mean_ms

                    # store every worker result before writing the final CPU rows
                    threaded_results: list[
                        tuple[int, TimingStatistics, ComparisonMetrics]
                    ] = []
                    best_cpu_ms = sequential_ms
                    best_cpu_output = sequential_output
                    best_cpu_backend = "Albumentations_Sequential"
                    best_cpu_workers = 1

                    # test all requested ThreadPool sizes for this workload
                    for workers in plan.thread_counts:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            threaded_operation = partial(
                                run_albumentations_threaded,
                                images,
                                prepared,
                                executor,
                            )
                            threaded_stats = measure(
                                threaded_operation,
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                            )
                        threaded_metrics = compare_batches(
                            sequential_output,
                            threaded_stats.output,
                        )
                        threaded_results.append(
                            (workers, threaded_stats, threaded_metrics)
                        )
                        # only an exact and faster result can replace the CPU reference
                        if (
                            threaded_metrics.exact_match
                            and threaded_stats.mean_ms < best_cpu_ms
                        ):
                            best_cpu_ms = threaded_stats.mean_ms
                            best_cpu_output = threaded_stats.output
                            best_cpu_backend = "Albumentations_ThreadPool"
                            best_cpu_workers = workers

                    # write the sequential row after the best CPU time is known
                    self_metrics = compare_batches(
                        sequential_output,
                        sequential_output,
                    )
                    _write_row(
                        writer,
                        file_handle,
                        _result_row(
                            base=base,
                            backend="Albumentations_Sequential",
                            library="Albumentations",
                            device="CPU",
                            workers=1,
                            parameter="SerialBatchLoop",
                            timer_scope="AugmentationOnly",
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                            statistics_data=sequential_stats,
                            sequential_ms=sequential_ms,
                            best_cpu_ms=best_cpu_ms,
                            reference_backend="Albumentations_Sequential",
                            comparison_kind="SelfReference",
                            metrics=self_metrics,
                            notes="OpenCV internal threads fixed to 1.",
                        ),
                    )

                    # write one CSV row for every measured worker count
                    for workers, threaded_stats, threaded_metrics in threaded_results:
                        _write_row(
                            writer,
                            file_handle,
                            _result_row(
                                base=base,
                                backend="Albumentations_ThreadPool",
                                library="Albumentations",
                                device="CPU",
                                workers=workers,
                                parameter=f"Threads={workers}",
                                timer_scope="AugmentationOnly_PoolReused",
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=threaded_stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                reference_backend="Albumentations_Sequential",
                                comparison_kind="SameLibrarySameParameters",
                                metrics=threaded_metrics,
                                notes=(
                                    "Thread-pool creation excluded; "
                                    "OpenCV internal threads fixed to 1."
                                ),
                            ),
                        )

                    print(
                        f"  Best CPU: {best_cpu_backend}, "
                        f"workers={best_cpu_workers}, {best_cpu_ms:.3f} ms"
                    )

                    # Kornia CPU uses one pre-converted BCHW tensor
                    cpu_tensor = numpy_batch_to_tensor(images)
                    cpu_parameters = prepare_kornia_parameters(
                        profile,
                        batch_size,
                        resolution,
                        resolution,
                        cpu_device,
                    )
                    kornia_cpu_operation = partial(
                        apply_kornia,
                        cpu_tensor,
                        cpu_parameters,
                    )
                    kornia_cpu_stats = measure(
                        kornia_cpu_operation,
                        warmups=plan.warmups,
                        repetitions=plan.repetitions,
                    )
                    kornia_cpu_output = tensor_to_numpy_batch(
                        kornia_cpu_stats.output
                    )
                    kornia_cpu_metrics = compare_batches(
                        best_cpu_output,
                        kornia_cpu_output,
                    )
                    _write_row(
                        writer,
                        file_handle,
                        _result_row(
                            base=base,
                            backend="Kornia_CPU_Batch",
                            library="Kornia/PyTorch",
                            device="CPU",
                            workers=1,
                            parameter="VectorizedBatch",
                            timer_scope="TensorAugmentationOnly",
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                            statistics_data=kornia_cpu_stats,
                            sequential_ms=sequential_ms,
                            best_cpu_ms=best_cpu_ms,
                            reference_backend=best_cpu_backend,
                            comparison_kind="CrossLibrarySameIntendedParameters",
                            metrics=kornia_cpu_metrics,
                            notes=(
                                "Tensor conversion excluded; "
                                "PyTorch CPU threads fixed to 1."
                            ),
                        ),
                    )

                    # keep two explicit skipped rows when CUDA is unavailable
                    if not cuda_available or cuda_device is None:
                        for backend, scope in (
                            (
                                "Kornia_CUDA_EndToEnd",
                                "HostToDevice+Augmentation+DeviceToHost",
                            ),
                            (
                                "Kornia_CUDA_DeviceOnly",
                                "DeviceAugmentationOnly",
                            ),
                        ):
                            _write_row(
                                writer,
                                file_handle,
                                _result_row(
                                    base=base,
                                    backend=backend,
                                    library="Kornia/PyTorch",
                                    device="CUDA",
                                    workers=None,
                                    parameter="VectorizedBatch",
                                    timer_scope=scope,
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics_data=None,
                                    sequential_ms=sequential_ms,
                                    best_cpu_ms=best_cpu_ms,
                                    reference_backend=best_cpu_backend,
                                    comparison_kind="SameWorkloadDifferentBackend",
                                    metrics=None,
                                    status="SKIPPED_NO_CUDA",
                                    notes=(
                                        "torch.cuda.is_available() returned False."
                                    ),
                                ),
                            )
                        print("  Kornia CUDA skipped: no CUDA device detected.")
                        continue

                    try:
                        # clear cached blocks and reset peak-memory tracking
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                        gpu_parameters = prepare_kornia_parameters(
                            profile,
                            batch_size,
                            resolution,
                            resolution,
                            cuda_device,
                        )

                        # end-to-end timing includes both transfers and augmentation
                        def run_cuda_end_to_end() -> torch.Tensor:
                            gpu_input = cpu_tensor.to(cuda_device)
                            gpu_output = apply_kornia(gpu_input, gpu_parameters)
                            return gpu_output.to(cpu_device)

                        cuda_e2e_stats = measure(
                            run_cuda_end_to_end,
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                            synchronize=torch.cuda.synchronize,
                        )
                        cuda_e2e_output = tensor_to_numpy_batch(
                            cuda_e2e_stats.output
                        )
                        cuda_e2e_metrics = compare_batches(
                            best_cpu_output,
                            cuda_e2e_output,
                        )
                        peak_e2e_memory = (
                            torch.cuda.max_memory_allocated(cuda_device)
                            / (1024.0**2)
                        )
                        _write_row(
                            writer,
                            file_handle,
                            _result_row(
                                base=base,
                                backend="Kornia_CUDA_EndToEnd",
                                library="Kornia/PyTorch",
                                device="CUDA",
                                workers=None,
                                parameter="VectorizedBatch",
                                timer_scope=(
                                    "HostToDevice+Augmentation+DeviceToHost"
                                ),
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=cuda_e2e_stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                reference_backend=best_cpu_backend,
                                comparison_kind=(
                                    "CrossLibrarySameIntendedParameters"
                                ),
                                metrics=cuda_e2e_metrics,
                                peak_memory_mb=peak_e2e_memory,
                                notes=(
                                    "Complete CUDA path directly compared "
                                    "with the fastest valid CPU result."
                                ),
                            ),
                        )

                        # device-only timing reuses an input already stored on the GPU
                        gpu_input = cpu_tensor.to(cuda_device)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                        cuda_device_operation = partial(
                            apply_kornia,
                            gpu_input,
                            gpu_parameters,
                        )
                        cuda_device_stats = measure(
                            cuda_device_operation,
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                            synchronize=torch.cuda.synchronize,
                        )
                        cuda_device_output = tensor_to_numpy_batch(
                            cuda_device_stats.output
                        )
                        cuda_device_metrics = compare_batches(
                            kornia_cpu_output,
                            cuda_device_output,
                        )
                        peak_device_memory = (
                            torch.cuda.max_memory_allocated(cuda_device)
                            / (1024.0**2)
                        )
                        _write_row(
                            writer,
                            file_handle,
                            _result_row(
                                base=base,
                                backend="Kornia_CUDA_DeviceOnly",
                                library="Kornia/PyTorch",
                                device="CUDA",
                                workers=None,
                                parameter="VectorizedBatch",
                                timer_scope="DeviceAugmentationOnly",
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=cuda_device_stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                reference_backend="Kornia_CPU_Batch",
                                comparison_kind="SameLibraryDifferentDevice",
                                metrics=cuda_device_metrics,
                                peak_memory_mb=peak_device_memory,
                                notes=(
                                    "Input already resident on the GPU; "
                                    "transfer costs excluded."
                                ),
                            ),
                        )
                        print(
                            f"  Kornia CUDA E2E: "
                            f"{cuda_e2e_stats.mean_ms:.3f} ms, "
                            f"best-CPU/GPU={best_cpu_ms / cuda_e2e_stats.mean_ms:.2f}x"
                        )
                        print(
                            f"  Kornia CUDA device-only: "
                            f"{cuda_device_stats.mean_ms:.3f} ms"
                        )
                    except RuntimeError as error:
                        # only CUDA out-of-memory errors are converted to skipped rows
                        if "out of memory" not in str(error).lower():
                            raise
                        torch.cuda.empty_cache()
                        for backend, scope in (
                            (
                                "Kornia_CUDA_EndToEnd",
                                "HostToDevice+Augmentation+DeviceToHost",
                            ),
                            (
                                "Kornia_CUDA_DeviceOnly",
                                "DeviceAugmentationOnly",
                            ),
                        ):
                            _write_row(
                                writer,
                                file_handle,
                                _result_row(
                                    base=base,
                                    backend=backend,
                                    library="Kornia/PyTorch",
                                    device="CUDA",
                                    workers=None,
                                    parameter="VectorizedBatch",
                                    timer_scope=scope,
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics_data=None,
                                    sequential_ms=sequential_ms,
                                    best_cpu_ms=best_cpu_ms,
                                    reference_backend=best_cpu_backend,
                                    comparison_kind="SameWorkloadDifferentBackend",
                                    metrics=None,
                                    status="SKIPPED_CUDA_OOM",
                                    notes=str(error).replace("\n", " "),
                                ),
                            )
                        print("  Kornia CUDA skipped: insufficient GPU memory.")

    print(f"\nBenchmark completed. CSV saved to:\n{result_path}")
    return result_path
