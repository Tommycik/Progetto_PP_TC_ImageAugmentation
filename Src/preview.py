from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import torch

from .albumentations_backend import prepare_albumentations, run_sequential, run_threaded
from .config import PREVIEWS_ROOT, PROFILES
from .cupy_backend import (
    apply_cupy_device,
    create_workspace,
    cupy_cuda_available,
    cupy_to_numpy_batch,
    numpy_batch_to_cupy,
    prepare_cupy_parameters,
    synchronize_cupy,
)
from .image_io import build_batch, make_contact_sheet, save_rgb_image
from .kornia_backend import (
    apply_kornia,
    numpy_batch_to_tensor,
    prepare_kornia_parameters,
    tensor_to_numpy_batch,
)
from .metrics import compare_batches


def generate_previews(
    source_images: list[np.ndarray],
    profile_name: str,
    resolution: int,
    batch_size: int,
) -> Path:
    profile = PROFILES[profile_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = PREVIEWS_ROOT / f"{timestamp}_{profile_name}_{resolution}_batch{batch_size}"
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Preview directory created:\n{output_directory}")

    images = build_batch(source_images, resolution, batch_size)
    prepared = prepare_albumentations(profile, batch_size)
    sequential = run_sequential(images, prepared)

    worker_count = min(12, max(1, batch_size))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        threaded = run_threaded(images, prepared, executor)

    cpu_tensor = numpy_batch_to_tensor(images)
    cpu_parameters = prepare_kornia_parameters(
        profile,
        batch_size,
        resolution,
        resolution,
        torch.device("cpu"),
    )
    kornia_cpu_tensor = apply_kornia(cpu_tensor, cpu_parameters)
    kornia_cpu = tensor_to_numpy_batch(kornia_cpu_tensor)

    groups: list[tuple[str, list[np.ndarray]]] = [
        ("Input", images),
        ("Albumentations sequential", sequential),
        (f"Albumentations {worker_count} threads", threaded),
        ("Kornia CPU", kornia_cpu),
    ]

    if cupy_cuda_available():
        cupy_parameters = prepare_cupy_parameters(
            profile, batch_size, resolution, resolution
        )
        cupy_input = numpy_batch_to_cupy(images)
        cupy_workspace = create_workspace(batch_size, resolution, resolution)
        cupy_output = apply_cupy_device(
            cupy_input, cupy_parameters, cupy_workspace, (16, 16)
        )
        synchronize_cupy()
        cupy_images = cupy_to_numpy_batch(cupy_output)
        groups.append(("CuPy CUDA block 16x16", cupy_images))
    else:
        cupy_images = []

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_parameters = prepare_kornia_parameters(
            profile,
            batch_size,
            resolution,
            resolution,
            device,
        )
        gpu_output = apply_kornia(cpu_tensor.to(device), gpu_parameters)
        torch.cuda.synchronize()
        kornia_cuda = tensor_to_numpy_batch(gpu_output)
        groups.append(("Kornia CUDA", kornia_cuda))
    else:
        kornia_cuda = []

    for group_name, group_images in groups:
        safe_name = group_name.lower().replace(" ", "_")
        for index, image in enumerate(group_images):
            save_rgb_image(output_directory / safe_name / f"image_{index:02d}.png", image)

    make_contact_sheet(groups, output_directory / "contact_sheet.png", max_images_per_group=batch_size)

    threaded_metrics = compare_batches(sequential, threaded)
    cross_cpu_metrics = compare_batches(sequential, kornia_cpu)
    print(f"Albumentations sequential/thread exact match: {threaded_metrics.exact_match}")
    print(
        "Albumentations/Kornia CPU: "
        f"MAE={cross_cpu_metrics.mae:.4f}, "
        f"SSIM={cross_cpu_metrics.global_ssim:.6f}"
    )
    if kornia_cuda:
        device_metrics = compare_batches(kornia_cpu, kornia_cuda)
        print(
            "Kornia CPU/CUDA: "
            f"exact={device_metrics.exact_match}, "
            f"MAE={device_metrics.mae:.4f}, "
            f"SSIM={device_metrics.global_ssim:.6f}"
        )

    if cupy_images:
        cupy_metrics = compare_batches(sequential, cupy_images)
        print(
            "Albumentations/CuPy block 16x16: "
            f"MAE={cupy_metrics.mae:.4f}, "
            f"SSIM={cupy_metrics.global_ssim:.6f}"
        )

    print(f"All preview images and the contact sheet were saved to:\n{output_directory}")
    if os.name == "nt":
        try:
            os.startfile(output_directory)
        except OSError:
            pass
    return output_directory
