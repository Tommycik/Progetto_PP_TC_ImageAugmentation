from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

# Image loading, resizing and preview-writing helpers.


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


def synthetic_image(width: int = 1024, height: int = 1024) -> np.ndarray:
    # Create a deterministic synthetic RGB image rich in geometric detail.
    y, x = np.mgrid[0:height, 0:width]
    red = ((x / max(width - 1, 1)) * 255.0).astype(np.uint8)
    green = ((y / max(height - 1, 1)) * 255.0).astype(np.uint8)
    blue = (((np.sin(x / 28.0) + np.cos(y / 35.0)) * 0.25 + 0.5) * 255.0).astype(np.uint8)
    image = np.stack([red, green, blue], axis=-1)

    pil_image = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle((width * 0.08, height * 0.10, width * 0.40, height * 0.34), outline="white", width=8)
    draw.ellipse((width * 0.56, height * 0.16, width * 0.88, height * 0.48), outline="black", width=10)
    draw.line((0, height - 1, width - 1, 0), fill="yellow", width=7)
    return np.asarray(pil_image, dtype=np.uint8)


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_source_images(path_text: str) -> tuple[list[np.ndarray], str]:
    # Load one image or a directory of images or use the synthetic fallback.
    cleaned = path_text.strip().strip('"')
    if not cleaned:
        return [synthetic_image()], "synthetic"

    path = Path(cleaned).expanduser().resolve()
    if path.is_file():
        return [_read_image(path)], str(path)

    if path.is_dir():
        candidates = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            raise ValueError(f"No supported images were found in: {path}")
        return [_read_image(item) for item in candidates], str(path)

    raise FileNotFoundError(f"Input path does not exist: {path}")


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    # Resize an image to a square resolution used by the benchmark.
    pil_image = Image.fromarray(image, mode="RGB")
    resized = pil_image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def build_batch(images: list[np.ndarray], size: int, batch_size: int) -> list[np.ndarray]:
    # Build a repeated batch from the available source images.
    if not images:
        raise ValueError("At least one source image is required.")
    resized = [resize_image(image, size) for image in images]
    return [resized[index % len(resized)].copy() for index in range(batch_size)]


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    # Save one RGB uint8 image to disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def make_contact_sheet(
    groups: Iterable[tuple[str, list[np.ndarray]]],
    output_path: Path,
    max_images_per_group: int = 8,
) -> None:
    # Create a simple contact sheet for quick visual comparison.
    prepared = [(name, images[:max_images_per_group]) for name, images in groups if images]
    if not prepared:
        return

    tile_width = 256
    tile_height = 256
    label_height = 34
    columns = max(len(images) for _, images in prepared)
    rows = len(prepared)
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, (name, images) in enumerate(prepared):
        y_offset = row_index * (tile_height + label_height)
        draw.text((8, y_offset + 8), name, fill="black")
        for column_index, image in enumerate(images):
            tile = Image.fromarray(image, mode="RGB").resize((tile_width, tile_height), Image.Resampling.BILINEAR)
            canvas.paste(tile, (column_index * tile_width, y_offset + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
