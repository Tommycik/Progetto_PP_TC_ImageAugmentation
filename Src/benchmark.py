from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
from pathlib import Path
from typing import Callable

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
from .config import BenchmarkPlan, PROFILES, RESULTS_ROOT
from .metrics import (
    ComparisonMetrics,
    TimingStatistics,
    compare_batches,
    measure,
    measure_paired_scopes,
    measure_torch_cpu,
)
from .report import (
    CSV_FIELDS,
    _base_row,
    _result_row,
    _timestamp,
    _write_row,
)


# complete benchmark execution

def run_benchmark(source_images: list[np.ndarray], input_source: str, plan: BenchmarkPlan, build_batch: Callable[[list[np.ndarray], int, int], list[np.ndarray]]) -> Path:
    # Execute every profile, resolution, batch and backend configuration.
    cv2.setNumThreads(1)
    result_path = (
        RESULTS_ROOT
        / f"augmentation_benchmark_{plan.name.lower()}_{_timestamp()}.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Kornia uses the same operators on CPU and CUDA.
    cpu_device = torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device("cuda:0") if cuda_available else None
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

                    # Prepare the common input before any timed backend call.
                    images = build_batch(source_images, resolution, batch_size)
                    prepared = prepare_albumentations(profile, batch_size)
                    base = _base_row(plan, input_source, profile_name, resolution, batch_size)

                    # Albumentations sequential is the stable image reference.
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

                    # Measure every Albumentations ThreadPool worker count.
                    threaded_results: list[
                        tuple[int, TimingStatistics, ComparisonMetrics]
                    ] = []
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

                    # Convert the batch once because Kornia's timing covers only tensor augmentation, exactly like the previous CPU scope.
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

                    # PyTorch Timer controls the intra-operation thread pool for
                    # each Kornia CPU measurement without changing the algorithm.
                    kornia_cpu_results: list[
                        tuple[
                            int,
                            TimingStatistics,
                            list[np.ndarray],
                            ComparisonMetrics,
                        ]
                    ] = []
                    for threads in plan.thread_counts:
                        kornia_stats = measure_torch_cpu(
                            kornia_cpu_operation,
                            threads=threads,
                            warmups=plan.warmups,
                            repetitions=plan.repetitions,
                        )
                        kornia_output = tensor_to_numpy_batch(kornia_stats.output)
                        kornia_metrics = compare_batches(
                            sequential_output,
                            kornia_output,
                        )
                        kornia_cpu_results.append(
                            (threads, kornia_stats, kornia_output, kornia_metrics)
                        )

                    # The one-thread Kornia row is its own scaling baseline.
                    kornia_one_thread = next(
                        (
                            result
                            for result in kornia_cpu_results
                            if result[0] == 1
                        ),
                        None,
                    )
                    if kornia_one_thread is None:
                        raise ValueError(
                            "Kornia CPU scaling requires thread_counts to include 1."
                        )
                    kornia_one_thread_ms = kornia_one_thread[1].mean_ms
                    kornia_reference_output = kornia_one_thread[2]

                    # Select the fastest correct CPU result across both libraries.
                    best_cpu_ms = sequential_ms
                    best_cpu_output = sequential_output
                    best_cpu_backend = "Albumentations_Sequential"
                    best_cpu_workers = 1

                    for workers, stats, metrics in threaded_results:
                        if metrics.exact_match and stats.mean_ms < best_cpu_ms:
                            best_cpu_ms = stats.mean_ms
                            best_cpu_output = stats.output
                            best_cpu_backend = "Albumentations_ThreadPool"
                            best_cpu_workers = workers

                    for threads, stats, output, metrics in kornia_cpu_results:
                        if metrics.tolerance_match and stats.mean_ms < best_cpu_ms:
                            best_cpu_ms = stats.mean_ms
                            best_cpu_output = output
                            best_cpu_backend = "Kornia_CPU_Batch"
                            best_cpu_workers = threads

                    # Write the CPU rows only after the global best CPU is known.
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
                            parallel_baseline_ms=sequential_ms,
                            reference_backend="Albumentations_Sequential",
                            comparison_kind="SelfReference",
                            metrics=self_metrics,
                            notes="OpenCV internal threads fixed to 1.",
                        ),
                    )

                    for workers, stats, metrics in threaded_results:
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
                                statistics_data=stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                parallel_baseline_ms=sequential_ms,
                                reference_backend="Albumentations_Sequential",
                                comparison_kind="SameLibrarySameParameters",
                                metrics=metrics,
                                notes=(
                                    "Thread-pool creation excluded; "
                                    "OpenCV internal threads fixed to 1."
                                ),
                            ),
                        )

                    for threads, stats, _output, metrics in kornia_cpu_results:
                        _write_row(
                            writer,
                            file_handle,
                            _result_row(
                                base=base,
                                backend="Kornia_CPU_Batch",
                                library="Kornia/PyTorch",
                                device="CPU",
                                workers=threads,
                                parameter=f"IntraOpThreads={threads}",
                                timer_scope="TensorAugmentationOnly",
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                parallel_baseline_ms=kornia_one_thread_ms,
                                reference_backend="Albumentations_Sequential",
                                comparison_kind="CrossLibrarySameIntendedParameters",
                                metrics=metrics,
                                notes=(
                                    "Tensor conversion excluded; PyTorch Timer "
                                    "sets the intra-operation thread count; "
                                    "inter-operation threads fixed to 1."
                                ),
                            ),
                        )

                    print(
                        f"  Best CPU: {best_cpu_backend}, "
                        f"threads={best_cpu_workers}, {best_cpu_ms:.3f} ms"
                    )

                    # CUDA executes the same Kornia operators once per repetition.
                    # The two CSV rows below are two timing scopes of this one GPU
                    # pipeline, not two different augmentation implementations.
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
                                    parameter="SameExecutionPairedTimers",
                                    timer_scope=scope,
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics_data=None,
                                    sequential_ms=sequential_ms,
                                    best_cpu_ms=best_cpu_ms,
                                    parallel_baseline_ms=None,
                                    reference_backend=best_cpu_backend,
                                    comparison_kind="SameWorkloadDifferentDevice",
                                    metrics=None,
                                    status="SKIPPED_NO_CUDA",
                                    notes="torch.cuda.is_available() returned False.",
                                ),
                            )
                        print("  Kornia CUDA skipped: no CUDA device detected.")
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

                        # One execution produces both timing scopes.
                        # The host timer covers transfers and augmentation.
                        # CUDA events delimit only the augmentation kernels.
                        def run_cuda_paired() -> tuple[torch.Tensor, float]:
                            start_event = torch.cuda.Event(
                                enable_timing=True
                            )
                            end_event = torch.cuda.Event(
                                enable_timing=True
                            )

                            gpu_input = cpu_tensor.to(cuda_device)
                            start_event.record()
                            gpu_output = apply_kornia(
                                gpu_input,
                                gpu_parameters,
                            )
                            end_event.record()
                            cpu_output = gpu_output.to(cpu_device)

                            torch.cuda.synchronize()
                            device_ms = start_event.elapsed_time(end_event)
                            return cpu_output, device_ms

                        cuda_e2e_stats, cuda_device_stats = (
                            measure_paired_scopes(
                                run_cuda_paired,
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                            )
                        )

                        cuda_output = tensor_to_numpy_batch(
                            cuda_e2e_stats.output
                        )
                        cuda_e2e_metrics = compare_batches(
                            best_cpu_output,
                            cuda_output,
                        )
                        cuda_device_metrics = compare_batches(
                            kornia_reference_output,
                            cuda_output,
                        )

                        peak_cuda_memory = (
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
                                parameter="SameExecutionPairedTimers",
                                timer_scope=(
                                    "HostToDevice+Augmentation+DeviceToHost"
                                ),
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=cuda_e2e_stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                parallel_baseline_ms=None,
                                reference_backend=best_cpu_backend,
                                comparison_kind="CompletePathVsBestCPU",
                                metrics=cuda_e2e_metrics,
                                peak_memory_mb=peak_cuda_memory,
                                notes=(
                                    "Host timer and CUDA events measure the "
                                    "same pipeline executions."
                                ),
                            ),
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
                                parameter="SameExecutionPairedTimers",
                                timer_scope="DeviceAugmentationOnly",
                                warmups=plan.warmups,
                                repetitions=plan.repetitions,
                                statistics_data=cuda_device_stats,
                                sequential_ms=sequential_ms,
                                best_cpu_ms=best_cpu_ms,
                                parallel_baseline_ms=None,
                                reference_backend="Kornia_CPU_Batch_1Thread",
                                comparison_kind=(
                                    "SameLibrarySamePipelineDifferentDevice"
                                ),
                                metrics=cuda_device_metrics,
                                peak_memory_mb=peak_cuda_memory,
                                notes=(
                                    "CUDA events delimit augmentation inside "
                                    "the same end-to-end executions."
                                ),
                            ),
                        )

                        print(
                            f"  Kornia CUDA paired timing, E2E scope: "
                            f"{cuda_e2e_stats.mean_ms:.3f} ms"
                        )
                        print(
                            f"  Kornia CUDA paired timing, device-only scope: "
                            f"{cuda_device_stats.mean_ms:.3f} ms"
                        )
                    except RuntimeError as error:
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
                                    parameter="SameExecutionPairedTimers",
                                    timer_scope=scope,
                                    warmups=plan.warmups,
                                    repetitions=plan.repetitions,
                                    statistics_data=None,
                                    sequential_ms=sequential_ms,
                                    best_cpu_ms=best_cpu_ms,
                                    parallel_baseline_ms=None,
                                    reference_backend=best_cpu_backend,
                                    comparison_kind="SameWorkloadDifferentDevice",
                                    metrics=None,
                                    status="SKIPPED_CUDA_OOM",
                                    notes=str(error).replace("\n", " "),
                                ),
                            )
                        print("  Kornia CUDA skipped: insufficient GPU memory.")

    print(f"\nBenchmark completed. CSV saved to:\n{result_path}")
    return result_path
