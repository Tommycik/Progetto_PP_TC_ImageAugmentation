from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime
from functools import partial
import math
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from .albumentations_backend import (
    prepare_albumentations,
    run_sequential,
    run_threaded,
)
from .config import BenchmarkPlan, PROFILES, RESULTS_ROOT
from .cupy_backend import (
    apply_cupy_device,
    clear_cupy_memory,
    create_workspace,
    cupy_cuda_available,
    cupy_device_name,
    cupy_to_host_float,
    host_float_to_cupy,
    host_float_to_numpy_batch,
    launch_geometry,
    numpy_float_batch,
    prepare_cupy_parameters,
    synchronize_cupy,
    used_memory_mb,
)
from .image_io import build_batch
from .kornia_backend import (
    apply_kornia,
    numpy_batch_to_tensor,
    prepare_kornia_parameters,
    tensor_to_numpy_batch,
)
from .metrics import ComparisonMetrics, compare_batches
from .timing import TimingStatistics, measure


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
    "BlockX",
    "BlockY",
    "ThreadsPerBlock",
    "GridX",
    "GridY",
    "GridZ",
    "LaunchedThreads",
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
    "Speedup",
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
    "BlockReferenceExactMatch",
    "BlockReferenceDifferentValues",
    "BlockReferenceMaxDifference",
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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _statistics_values(statistics: TimingStatistics[Any] | None) -> dict[str, str]:
    if statistics is None:
        return {
            "TimeMean_ms": "",
            "TimeMedian_ms": "",
            "TimeMin_ms": "",
            "TimeMax_ms": "",
            "StdDev_ms": "",
            "CoefficientVariation_percent": "",
        }
    return {
        "TimeMean_ms": _format_number(statistics.mean_ms),
        "TimeMedian_ms": _format_number(statistics.median_ms),
        "TimeMin_ms": _format_number(statistics.min_ms),
        "TimeMax_ms": _format_number(statistics.max_ms),
        "StdDev_ms": _format_number(statistics.stddev_ms),
        "CoefficientVariation_percent": _format_number(
            statistics.coefficient_variation_percent
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
    statistics: TimingStatistics[Any] | None,
    baseline_ms: float | None,
    reference_backend: str,
    comparison_kind: str,
    metrics: ComparisonMetrics | None,
    peak_memory_mb: float | None = None,
    block_geometry: Any | None = None,
    block_reference_metrics: ComparisonMetrics | None = None,
    status: str = "SUCCESS",
    notes: str = "",
) -> dict[str, str]:
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
            "SequentialBaseline_ms": _format_number(baseline_ms),
            "ReferenceBackend": reference_backend,
            "ComparisonKind": comparison_kind,
            "CudaPeakMemory_MB": _format_number(peak_memory_mb),
            "Status": status,
            "Notes": notes,
        }
    )
    row.update(_statistics_values(statistics))
    row.update(_metric_values(metrics))

    if block_geometry is not None:
        row.update(
            {
                "BlockX": str(block_geometry.block_x),
                "BlockY": str(block_geometry.block_y),
                "ThreadsPerBlock": str(block_geometry.threads_per_block),
                "GridX": str(block_geometry.grid_x),
                "GridY": str(block_geometry.grid_y),
                "GridZ": str(block_geometry.grid_z),
                "LaunchedThreads": str(block_geometry.launched_threads),
            }
        )

    if block_reference_metrics is not None:
        row.update(
            {
                "BlockReferenceExactMatch": (
                    "YES" if block_reference_metrics.exact_match else "NO"
                ),
                "BlockReferenceDifferentValues": str(
                    block_reference_metrics.different_values
                ),
                "BlockReferenceMaxDifference": _format_number(
                    block_reference_metrics.max_difference
                ),
            }
        )

    if statistics is not None and statistics.mean_ms > 0.0:
        batch_size = int(base["BatchSize"])
        resolution = int(base["Resolution"].split("x", maxsplit=1)[0])
        speedup = baseline_ms / statistics.mean_ms if baseline_ms else 1.0
        row["Speedup"] = _format_number(speedup)
        row["Throughput_images_s"] = _format_number(batch_size * 1000.0 / statistics.mean_ms)
        row["Throughput_MPixels_s"] = _format_number(
            batch_size * resolution * resolution / statistics.mean_ms / 1000.0
        )
        if workers is not None and workers > 0:
            row["ParallelEfficiency"] = _format_number(speedup / workers)

    return row


def _write_row(writer: csv.DictWriter, file_handle: Any, row: dict[str, str]) -> None:
    writer.writerow(row)
    file_handle.flush()


def run_benchmark(
    source_images: list[np.ndarray],
    input_source: str,
    plan: BenchmarkPlan,
) -> Path:
    cv2.setNumThreads(1)
    result_path = RESULTS_ROOT / f"augmentation_benchmark_{plan.name.lower()}_{_timestamp()}.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device("cuda:0") if cuda_available else None
    cpu_device = torch.device("cpu")

    total_workloads = sum(len(batch_sizes) for _, batch_sizes in plan.workloads) * len(plan.profiles)
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
                        f"Profile={profile_name}, Resolution={resolution}x{resolution}, "
                        f"Batch={batch_size}"
                    )

                    images = build_batch(source_images, resolution, batch_size)
                    prepared_alb = prepare_albumentations(profile, batch_size)
                    base = _base_row(plan, input_source, profile_name, resolution, batch_size)

                    sequential_stats = measure(
                        lambda: run_sequential(images, prepared_alb),
                        warmups=plan.warmups,
                        repetitions=plan.repetitions,
                    )
                    sequential_output = sequential_stats.output
                    self_metrics = compare_batches(sequential_output, sequential_output)
                    baseline_ms = sequential_stats.mean_ms
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
                            statistics=sequential_stats,
                            baseline_ms=baseline_ms,
                            reference_backend="Albumentations_Sequential",
                            comparison_kind="SelfReference",
                            metrics=self_metrics,
                            notes="OpenCV internal threads fixed to 1.",
                        ),
                    )
                    print(f"  Sequential: {baseline_ms:.3f} ms")

                    for workers in plan.thread_counts:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            threaded_operation = partial(
                                run_threaded,
                                images,
                                prepared_alb,
                                executor,
                            )
                            threaded_stats = measure(
                                threaded_operation,
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                            )
                        threaded_metrics = compare_batches(sequential_output, threaded_stats.output)
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
                                statistics=threaded_stats,
                                baseline_ms=baseline_ms,
                                reference_backend="Albumentations_Sequential",
                                comparison_kind="SameLibrarySameParameters",
                                metrics=threaded_metrics,
                                notes="ThreadPool creation excluded; OpenCV internal threads fixed to 1.",
                            ),
                        )
                        print(
                            f"  Threads {workers:2d}: {threaded_stats.mean_ms:.3f} ms, "
                            f"speedup {baseline_ms / threaded_stats.mean_ms:.2f}x, "
                            f"exact={threaded_metrics.exact_match}"
                        )

                    cpu_tensor = numpy_batch_to_tensor(images)
                    cpu_parameters = prepare_kornia_parameters(
                        profile,
                        batch_size,
                        resolution,
                        resolution,
                        cpu_device,
                    )
                    kornia_cpu_stats = measure(
                        lambda: apply_kornia(cpu_tensor, cpu_parameters),
                        warmups=plan.warmups,
                        repetitions=plan.repetitions,
                    )
                    kornia_cpu_output = tensor_to_numpy_batch(kornia_cpu_stats.output)
                    kornia_cpu_metrics = compare_batches(sequential_output, kornia_cpu_output)
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
                            statistics=kornia_cpu_stats,
                            baseline_ms=baseline_ms,
                            reference_backend="Albumentations_Sequential",
                            comparison_kind="CrossLibrarySameIntendedParameters",
                            metrics=kornia_cpu_metrics,
                            notes="Tensor conversion excluded; PyTorch CPU threads fixed to 1.",
                        ),
                    )
                    print(
                        f"  Kornia CPU: {kornia_cpu_stats.mean_ms:.3f} ms, "
                        f"SSIM={kornia_cpu_metrics.global_ssim:.5f}"
                    )

                    # CuPy uses explicit RawKernel launches so block dimensions and
                    # threads per block can be benchmarked directly from Python.
                    reference_block = (16, 16) if (16, 16) in plan.gpu_blocks else plan.gpu_blocks[0]
                    if not cupy_cuda_available():
                        for block_x, block_y in plan.gpu_blocks:
                            geometry = launch_geometry(
                                resolution, resolution, batch_size, (block_x, block_y)
                            )
                            _write_row(
                                writer,
                                file_handle,
                                _result_row(
                                    base=base,
                                    backend="CuPy_RawKernel_DeviceOnly",
                                    library="CuPy RawKernel",
                                    device="CUDA",
                                    workers=None,
                                    parameter=f"Block={block_x}x{block_y}",
                                    timer_scope="DeviceAugmentationOnly",
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics=None,
                                    baseline_ms=baseline_ms,
                                    reference_backend=f"CuPy_RawKernel_{reference_block[0]}x{reference_block[1]}",
                                    comparison_kind="SameKernelDifferentBlock",
                                    metrics=None,
                                    block_geometry=geometry,
                                    status="SKIPPED_NO_CUPY_CUDA",
                                    notes="CuPy is missing or does not detect a CUDA device.",
                                ),
                            )
                        print("  CuPy block tests skipped: CuPy CUDA is unavailable.")
                    else:
                        try:
                            clear_cupy_memory()
                            host_float_batch = numpy_float_batch(images)
                            cupy_parameters = prepare_cupy_parameters(
                                profile, batch_size, resolution, resolution
                            )

                            # End-to-end is measured once with the reference block.
                            # Transfer time does not depend on the block geometry.
                            reference_geometry = launch_geometry(
                                resolution, resolution, batch_size, reference_block
                            )

                            def run_cupy_end_to_end() -> np.ndarray:
                                device_input = host_float_to_cupy(host_float_batch)
                                workspace = create_workspace(
                                    batch_size, resolution, resolution
                                )
                                device_output = apply_cupy_device(
                                    device_input, cupy_parameters, workspace, reference_block
                                )
                                return cupy_to_host_float(device_output)

                            cupy_e2e_stats = measure(
                                run_cupy_end_to_end,
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                synchronize=synchronize_cupy,
                            )
                            cupy_e2e_output = host_float_to_numpy_batch(cupy_e2e_stats.output)
                            cupy_e2e_metrics = compare_batches(
                                sequential_output, cupy_e2e_output
                            )
                            _write_row(
                                writer,
                                file_handle,
                                _result_row(
                                    base=base,
                                    backend="CuPy_RawKernel_EndToEnd",
                                    library="CuPy RawKernel",
                                    device="CUDA",
                                    workers=None,
                                    parameter=(
                                        f"ReferenceBlock={reference_block[0]}x{reference_block[1]}"
                                    ),
                                    timer_scope=(
                                        "HostToDevice+RawKernels+DeviceToHost"
                                    ),
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics=cupy_e2e_stats,
                                    baseline_ms=baseline_ms,
                                    reference_backend="Albumentations_Sequential",
                                    comparison_kind=(
                                        "CrossImplementationSameIntendedParameters"
                                    ),
                                    metrics=cupy_e2e_metrics,
                                    peak_memory_mb=used_memory_mb(),
                                    block_geometry=reference_geometry,
                                    notes=(
                                        "End-to-end CuPy measurement uses the reference block. "
                                        "Block-size tests below exclude transfers."
                                    ),
                                ),
                            )

                            device_input = host_float_to_cupy(host_float_batch)
                            reference_workspace = create_workspace(
                                batch_size, resolution, resolution
                            )
                            reference_device_output = apply_cupy_device(
                                device_input,
                                cupy_parameters,
                                reference_workspace,
                                reference_block,
                            )
                            synchronize_cupy()
                            reference_output = host_float_to_numpy_batch(
                                cupy_to_host_float(reference_device_output)
                            )

                            for block_x, block_y in plan.gpu_blocks:
                                current_block = (block_x, block_y)
                                geometry = launch_geometry(
                                    resolution, resolution, batch_size, current_block
                                )
                                current_workspace = create_workspace(
                                    batch_size, resolution, resolution
                                )
                                cupy_operation = partial(
                                    apply_cupy_device,
                                    device_input,
                                    cupy_parameters,
                                    current_workspace,
                                    current_block,
                                )
                                cupy_device_stats = measure(
                                    cupy_operation,
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    synchronize=synchronize_cupy,
                                )
                                block_output = host_float_to_numpy_batch(
                                    cupy_to_host_float(cupy_device_stats.output)
                                )
                                cross_metrics = compare_batches(
                                    sequential_output, block_output
                                )
                                block_metrics = compare_batches(
                                    reference_output, block_output
                                )
                                _write_row(
                                    writer,
                                    file_handle,
                                    _result_row(
                                        base=base,
                                        backend="CuPy_RawKernel_DeviceOnly",
                                        library="CuPy RawKernel",
                                        device="CUDA",
                                        workers=None,
                                        parameter=f"Block={block_x}x{block_y}",
                                        timer_scope="DeviceRawKernelsOnly",
                                        warmups=plan.warmups,
                                        repetitions=plan.repetitions,
                                        statistics=cupy_device_stats,
                                        baseline_ms=baseline_ms,
                                        reference_backend=(
                                            f"CuPy_RawKernel_{reference_block[0]}x{reference_block[1]}"
                                        ),
                                        comparison_kind="SameKernelDifferentBlock",
                                        metrics=cross_metrics,
                                        peak_memory_mb=used_memory_mb(),
                                        block_geometry=geometry,
                                        block_reference_metrics=block_metrics,
                                        notes=(
                                            "Main metrics compare with Albumentations. "
                                            "BlockReference fields compare with the fixed CuPy reference block."
                                        ),
                                    ),
                                )
                                print(
                                    f"  CuPy block {block_x:3d}x{block_y:<3d} "
                                    f"({geometry.threads_per_block:4d} threads): "
                                    f"{cupy_device_stats.mean_ms:.3f} ms, "
                                    f"block_match={block_metrics.exact_match}"
                                )
                        except Exception as error:
                            clear_cupy_memory()
                            for block_x, block_y in plan.gpu_blocks:
                                geometry = launch_geometry(
                                    resolution, resolution, batch_size, (block_x, block_y)
                                )
                                _write_row(
                                    writer,
                                    file_handle,
                                    _result_row(
                                        base=base,
                                        backend="CuPy_RawKernel_DeviceOnly",
                                        library="CuPy RawKernel",
                                        device="CUDA",
                                        workers=None,
                                        parameter=f"Block={block_x}x{block_y}",
                                        timer_scope="DeviceRawKernelsOnly",
                                        warmups=plan.warmups,
                                        repetitions=plan.repetitions,
                                        statistics=None,
                                        baseline_ms=baseline_ms,
                                        reference_backend=(
                                            f"CuPy_RawKernel_{reference_block[0]}x{reference_block[1]}"
                                        ),
                                        comparison_kind="SameKernelDifferentBlock",
                                        metrics=None,
                                        block_geometry=geometry,
                                        status="ERROR_CUPY",
                                        notes=str(error).replace("\n", " "),
                                    ),
                                )
                            print(f"  CuPy block tests failed: {error}")

                    if not cuda_available or cuda_device is None:
                        for backend, scope in (
                            ("Kornia_CUDA_EndToEnd", "HostToDevice+Augmentation+DeviceToHost"),
                            ("Kornia_CUDA_DeviceOnly", "DeviceAugmentationOnly"),
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
                                    statistics=None,
                                    baseline_ms=baseline_ms,
                                    reference_backend="Kornia_CPU_Batch",
                                    comparison_kind="SameLibraryDifferentDevice",
                                    metrics=None,
                                    status="SKIPPED_NO_CUDA",
                                    notes="torch.cuda.is_available() returned False.",
                                ),
                            )
                        print("  CUDA skipped: PyTorch does not report an available CUDA device.")
                        continue

                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                        gpu_parameters = prepare_kornia_parameters(
                            profile,
                            batch_size,
                            resolution,
                            resolution,
                            cuda_device,
                        )

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
                        cuda_e2e_output = tensor_to_numpy_batch(cuda_e2e_stats.output)
                        cuda_e2e_metrics = compare_batches(sequential_output, cuda_e2e_output)
                        peak_memory_mb = torch.cuda.max_memory_allocated(cuda_device) / (1024.0 ** 2)
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
                                timer_scope="HostToDevice+Augmentation+DeviceToHost",
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics=cuda_e2e_stats,
                                baseline_ms=baseline_ms,
                                reference_backend="Albumentations_Sequential",
                                comparison_kind="CrossLibrarySameIntendedParameters",
                                metrics=cuda_e2e_metrics,
                                peak_memory_mb=peak_memory_mb,
                                notes="Directly comparable end-to-end CUDA scope.",
                            ),
                        )

                        gpu_input = cpu_tensor.to(cuda_device)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                        cuda_device_stats = measure(
                            lambda: apply_kornia(gpu_input, gpu_parameters),
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                            synchronize=torch.cuda.synchronize,
                        )
                        cuda_device_output = tensor_to_numpy_batch(cuda_device_stats.output)
                        cuda_device_metrics = compare_batches(kornia_cpu_output, cuda_device_output)
                        peak_device_memory_mb = torch.cuda.max_memory_allocated(cuda_device) / (1024.0 ** 2)
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
                                statistics=cuda_device_stats,
                                baseline_ms=baseline_ms,
                                reference_backend="Kornia_CPU_Batch",
                                comparison_kind="SameLibraryDifferentDevice",
                                metrics=cuda_device_metrics,
                                peak_memory_mb=peak_device_memory_mb,
                                notes="Input already on GPU; use separately from end-to-end speedup.",
                            ),
                        )
                        print(
                            f"  CUDA E2E: {cuda_e2e_stats.mean_ms:.3f} ms, "
                            f"speedup {baseline_ms / cuda_e2e_stats.mean_ms:.2f}x, "
                            f"SSIM={cuda_e2e_metrics.global_ssim:.5f}"
                        )
                        print(
                            f"  CUDA device: {cuda_device_stats.mean_ms:.3f} ms, "
                            f"CPU/GPU exact={cuda_device_metrics.exact_match}"
                        )
                    except RuntimeError as error:
                        if "out of memory" not in str(error).lower():
                            raise
                        torch.cuda.empty_cache()
                        for backend, scope in (
                            ("Kornia_CUDA_EndToEnd", "HostToDevice+Augmentation+DeviceToHost"),
                            ("Kornia_CUDA_DeviceOnly", "DeviceAugmentationOnly"),
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
                                    statistics=None,
                                    baseline_ms=baseline_ms,
                                    reference_backend="Kornia_CPU_Batch",
                                    comparison_kind="SameLibraryDifferentDevice",
                                    metrics=None,
                                    status="SKIPPED_CUDA_OOM",
                                    notes=str(error).replace("\n", " "),
                                ),
                            )
                        print("  CUDA skipped for this workload: insufficient GPU memory.")

    print(f"\nBenchmark completed. CSV saved to:\n{result_path}")
    return result_path