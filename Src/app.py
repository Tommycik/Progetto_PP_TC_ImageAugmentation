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
    FULL_PLAN,
    PREVIEWS_ROOT,
    PROFILES,
    QUICK_PLAN,
    RESULTS_ROOT,
    ensure_output_directories,
)
from .image_io import load_source_images
from .preview import generate_previews


def _configure_runtime() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _read_source_images() -> tuple[list[object], str]:
    print("\nEnter an image path or a directory containing images.")
    print("Press Enter to use the deterministic synthetic image.")
    path_text = input("Input path: ")
    images, source = load_source_images(path_text)
    print(f"Loaded {len(images)} source image(s) from: {source}")
    return images, source


def _select_profile() -> str:
    names = list(PROFILES.keys())
    print("\nAugmentation profiles:")
    for index, name in enumerate(names, start=1):
        print(f"{index}. {name}: {PROFILES[name].description}")
    while True:
        choice = input("Profile: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid profile selection.")


def _read_integer(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        if value.isdigit() and minimum <= int(value) <= maximum:
            return int(value)
        print(f"Enter a value between {minimum} and {maximum}.")


def _environment_report() -> None:
    print("\nENVIRONMENT")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"NumPy: {numpy.__version__}")
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
    ensure_output_directories()
    _configure_runtime()

    print("IMAGE AUGMENTATION BENCHMARK - PYTHON")
    print(
        "Albumentations sequential/threaded CPU, Kornia CPU/CUDA and "
        "CuPy CUDA block-size tests"
    )
    print(f"CSV output: {RESULTS_ROOT}")
    print(f"Preview output: {PREVIEWS_ROOT}")

    while True:
        print("\n1. Quick benchmark")
        print("2. Full benchmark")
        print("3. Generate previews")
        print("4. Check environment")
        print("5. Exit")
        choice = input("Selection: ").strip()

        try:
            if choice == "1":
                images, source = _read_source_images()
                run_benchmark(images, source, QUICK_PLAN)
            elif choice == "2":
                print(
                    "\nThe full benchmark tests five profiles, batches up to 32, "
                    "six CPU thread counts and nine explicit CUDA block layouts."
                )
                confirmation = input("Continue? [y/N]: ").strip().lower()
                if confirmation != "y":
                    continue
                images, source = _read_source_images()
                run_benchmark(images, source, FULL_PLAN)
            elif choice == "3":
                images, _ = _read_source_images()
                profile_name = _select_profile()
                resolution = _read_integer("Preview resolution", 512, 128, 2048)
                batch_size = _read_integer("Preview batch size", 8, 1, 32)
                generate_previews(images, profile_name, resolution, batch_size)
            elif choice == "4":
                _environment_report()
            elif choice == "5":
                print("Program closed.")
                return
            else:
                print("Invalid selection.")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            print(f"\nERROR: {error}")
