from __future__ import annotations

import os
import platform
import sys

import albumentations
import cv2
import kornia
import numpy
import torch
from PIL import Image

from .benchmark import run_benchmark
from .cupy_backend import (
    cupy_compute_capability,
    cupy_cuda_available,
    cupy_device_name,
    cupy_imported,
    cupy_version,
)
from .config import (
    FULL_BENCHMARK_PLAN,
    PREVIEWS_ROOT,
    PROFILES,
    RESULTS_ROOT,
    ensure_output_directories,
)
from .image_io import load_source_images
from .preview import generate_previews


def _configure_runtime() -> None:
    # Reduce hidden internal threading so the benchmark stays controlled.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing the setting before certain operations.
        pass


def _read_source_images() -> tuple[list[object], str]:
    # Read one image, a directory of images or fall back to the synthetic input.
    print("\nEnter an image path or a directory containing images.")
    print("Press Enter to use the deterministic synthetic image.")
    path_text = input("Input path: ")
    images, source = load_source_images(path_text)
    print(f"Loaded {len(images)} source image(s) from: {source}")
    return images, source

def _read_integer(prompt: str, default: int, minimum: int, maximum: int) -> int:
    # Read a bounded integer from the console.
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        if value.isdigit() and minimum <= int(value) <= maximum:
            return int(value)
        print(f"Enter a value between {minimum} and {maximum}.")


def _environment_report() -> None:
    # Print a compact runtime report useful before starting the benchmark.
    print("\nENVIRONMENT")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Pillow: {Image.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Albumentations: {albumentations.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Kornia: {kornia.__version__}")
    print(f"CuPy: {cupy_version()}")
    print(f"CuPy imported: {cupy_imported()}")
    print(f"CuPy CUDA available: {cupy_cuda_available()}")
    if cupy_cuda_available():
        print(f"CuPy GPU: {cupy_device_name()}")
        print(f"CuPy compute capability: {cupy_compute_capability()}")
    print(f"CUDA available through PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime used by PyTorch: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {properties.total_memory / (1024 ** 3):.2f} GiB")
    print(f"CSV directory: {RESULTS_ROOT}")
    print(f"Preview directory: {PREVIEWS_ROOT}")


def run_application() -> None:
    # Start the interactive menu and run the benchmark or generate previews.
    ensure_output_directories()
    _configure_runtime()

    print("Image augmentation benchmark and preview generator")
    print(
        "Albumentations sequential/threaded CPU, Kornia CPU/CUDA and "
        "CuPy CUDA block-size tests"
    )
    print(f"CSV output: {RESULTS_ROOT}")
    print(f"Preview output: {PREVIEWS_ROOT}")

    while True:
        print("\n1. Benchmark")
        print("2. Generate previews")
        print("3. Check environment")
        print("4. Exit")
        choice = input("Selection: ").strip()

        try:
            if choice == "1":
                images, source = _read_source_images()
                run_benchmark(images, source, FULL_BENCHMARK_PLAN)
            elif choice == "2":
                images, _ = _read_source_images()

                generate_previews(
                    images,
                    profile_name="MixedStrong",
                    resolution=512,
                    batch_size=8,
                    mode="fast",
                )
            elif choice == "3":
                _environment_report()
            elif choice == "4":
                print("Program closed.")
                return
            else:
                print("Invalid selection.")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            print(f"\nERROR: {error}")
