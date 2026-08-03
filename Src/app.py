from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import platform
import sys

import albumentations
import cv2
import kornia
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch

from .backends import (
    prepare_albumentations,
    run_albumentations_threaded,
)
from .benchmark import run_benchmark
from .config import (
    FULL_BENCHMARK_PLAN,
    PREVIEWS_ROOT,
    PROFILES,
    RESULTS_ROOT,
    ensure_output_directories,
)


# image formats accepted by the input loader
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".ppm",
    ".pgm",
}


# runtime configuration

def _configure_runtime() -> None:
    # one OpenCV thread prevents nested parallelism inside each worker
    cv2.setNumThreads(1)

    # one thread is the default outside measurements. The benchmark uses
    # PyTorch Timer to apply each configured intra-operation thread count.
    torch.set_num_threads(1)
    try:
        # keep inter-operation scheduling fixed while intra-operation threads vary
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may reject the change after its thread system is initialized.
        pass

# image loading and batch construction
def _synthetic_image(width: int = 1024, height: int = 1024) -> np.ndarray:
    # coordinate matrices used to build the three colour channels
    y, x = np.mgrid[0:height, 0:width]
    red = ((x / max(width - 1, 1)) * 255.0).astype(np.uint8)
    green = ((y / max(height - 1, 1)) * 255.0).astype(np.uint8)
    blue = (
        ((np.sin(x / 28.0) + np.cos(y / 35.0)) * 0.25 + 0.5) * 255.0
    ).astype(np.uint8)
    # combine the deterministic channels into one RGB image
    image = np.stack([red, green, blue], axis=-1)

    # add strong edges and shapes so rotations and blur remain visible
    pil_image = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(
        (width * 0.08, height * 0.10, width * 0.40, height * 0.34),
        outline="white",
        width=8,
    )
    draw.ellipse(
        (width * 0.56, height * 0.16, width * 0.88, height * 0.48),
        outline="black",
        width=10,
    )
    draw.line((0, height - 1, width - 1, 0), fill="yellow", width=7)
    return np.asarray(pil_image, dtype=np.uint8)


def _read_image(path: Path) -> np.ndarray:
    # convert every source to the same RGB uint8 representation
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_source_images(path_text: str) -> tuple[list[np.ndarray], str]:
    # Load one image, all images in a directory or the synthetic fallback.
    # remove spaces and quotes copied from Windows paths
    cleaned = path_text.strip().strip('"')
    if not cleaned:
        # empty input selects the deterministic synthetic image
        return [_synthetic_image()], "synthetic"

    path = Path(cleaned).expanduser().resolve()
    if path.is_file():
        # a single file produces a one-image source list
        return [_read_image(path)], str(path)

    if path.is_dir():
        # a directory loads every supported image in alphabetical order
        candidates = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            raise ValueError(f"No supported images were found in: {path}")
        return [_read_image(item) for item in candidates], str(path)

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _read_source_images() -> tuple[list[np.ndarray], str]:
    # read the path from the console and report the selected source
    print("\nEnter an image path or a directory containing images.")
    print("Press Enter to use the deterministic synthetic image.")
    images, source = _load_source_images(input("Input path: "))
    print(f"Loaded {len(images)} source image(s) from: {source}")
    return images, source


def _build_batch(images: list[np.ndarray], size: int, batch_size: int,) -> list[np.ndarray]:
    # resize every source once before constructing the repeated batch
    if not images:
        raise ValueError("At least one source image is required.")
    # all benchmark images use a square resolution
    resized = [
        np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (size, size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        for image in images
    ]
    # repeat the available sources when the requested batch is larger
    return [resized[index % len(resized)].copy() for index in range(batch_size)]


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    # create the backend folder before saving the preview image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(image, 0, 255).astype(np.uint8),
        mode="RGB",
    ).save(path)


def _make_contact_sheet(
    groups: list[tuple[str, list[np.ndarray]]],
    output_path: Path,
) -> None:
    # place the input and augmented batches in labelled rows
    tile_width = 256
    tile_height = 256
    label_height = 34
    # one column is reserved for every image in the largest group
    columns = max(len(images) for _, images in groups)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, len(groups) * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    # each group occupies one row of the final contact sheet
    for row_index, (name, images) in enumerate(groups):
        y_offset = row_index * (tile_height + label_height)
        draw.text((8, y_offset + 8), name, fill="black")
        for column_index, image in enumerate(images):
            tile = Image.fromarray(image, mode="RGB").resize(
                (tile_width, tile_height),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(
                tile,
                (column_index * tile_width, y_offset + label_height),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

# preview generation
def _generate_preview(source_images: list[np.ndarray]) -> Path:
    # one fixed and predictable preview configuration.
    profile_name = "MixedStrong"
    resolution = 512
    batch_size = 8
    worker_count = 8

    # Build the deterministic batch
    profile = PROFILES[profile_name]
    images = _build_batch(source_images, resolution, batch_size)
    prepared = prepare_albumentations(profile, batch_size)

    # Apply Albumentations with exactly eight CPU workers.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        preview_images = run_albumentations_threaded(
            images,
            prepared,
            executor,
        )

    print("\nPreview backend: Albumentations")
    print(f"CPU workers: {worker_count}")
    print(
        f"Parameters: profile={profile_name}, resolution={resolution}, "
        f"batch={batch_size}"
    )

    # Create a separate timestamped directory for every preview.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = (
        PREVIEWS_ROOT
        / f"{timestamp}_{profile_name}_{resolution}_batch{batch_size}"
    )

    # Save the original batch and the augmented result.
    for index, image in enumerate(images):
        _save_rgb_image(
            output_directory / "input" / f"image_{index:02d}.png",
            image,
        )
    for index, image in enumerate(preview_images):
        _save_rgb_image(
            output_directory / "albumentations_8_threads" / f"image_{index:02d}.png",
            image,
        )

    _make_contact_sheet(
        [("Input", images), ("Albumentations 8 threads", preview_images)],
        output_directory / "contact_sheet.png",
    )

    print(f"Preview saved to:\n{output_directory}")
    if os.name == "nt":
        try:
            os.startfile(output_directory)
        except OSError:
            pass
    return output_directory


# environment report and interactive menu
def _environment_report() -> None:
    # print software, CPU and GPU information used by the benchmark
    print("\nENVIRONMENT")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical CPU cores: {os.cpu_count()}")
    print(f"NumPy: {np.__version__}")
    print(f"Pillow: {PIL.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Albumentations: {albumentations.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch default intra-op threads: {torch.get_num_threads()}")
    print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"Kornia: {kornia.__version__}")
    print(f"CUDA available through PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime used by PyTorch: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {properties.total_memory / (1024**3):.2f} GiB")
    print(f"CSV directory: {RESULTS_ROOT}")
    print(f"Preview directory: {PREVIEWS_ROOT}")


def run_application() -> None:
    # create output folders and start the menu without command-line arguments
    ensure_output_directories()
    _configure_runtime()

    print("Image augmentation benchmark")
    print("Albumentations CPU versus Kornia CPU/CUDA")
    print(f"CSV output: {RESULTS_ROOT}")
    print(f"Preview output: {PREVIEWS_ROOT}")

    while True:
        print("\n1. Full benchmark")
        print("2. Generate preview")
        print("3. Check environment")
        print("4. Exit")
        choice = input("Selection: ").strip()

        try:
            if choice == "1":
                # run the complete workload matrix and write one CSV file
                images, source = _read_source_images()
                run_benchmark(
                    images,
                    source,
                    FULL_BENCHMARK_PLAN,
                    _build_batch,
                )
            elif choice == "2":
                # generate one fixed Albumentations preview with eight workers
                images, _ = _read_source_images()
                _generate_preview(images)
            elif choice == "3":
                # show the runtime configuration before a long benchmark
                _environment_report()
            elif choice == "4":
                print("Program closed.")
                return
            else:
                print("Invalid selection.")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            print(f"\nERROR: {error}")
