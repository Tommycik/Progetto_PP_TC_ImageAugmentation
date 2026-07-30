# CuPy RawKernel backend.
#
# This module exists because Kornia does not expose low-level CUDA block sizes.
# CuPy RawKernel makes it possible to keep the project in Python while still
# measuring the effect of explicit block geometries on GPU execution.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import AugmentationProfile
from .profiles import parameters_for_sample

try:
    import cupy as cp
except ImportError:  # CuPy remains optional so CPU-only startup can report the issue.
    cp = None  # type: ignore[assignment]


# CUDA kernels used by the CuPy backend.
AFFINE_COLOR_SOURCE = r'''
extern "C" {

__device__ __forceinline__ int reflect101(int coordinate, int length) {
    if (length <= 1) return 0;
    while (coordinate < 0 || coordinate >= length) {
        if (coordinate < 0) {
            coordinate = -coordinate;
        } else {
            coordinate = 2 * length - coordinate - 2;
        }
    }
    return coordinate;
}

__device__ __forceinline__ float read_rgb(
    const float* input,
    int batch_index,
    int y,
    int x,
    int channel,
    int height,
    int width
) {
    const int yy = reflect101(y, height);
    const int xx = reflect101(x, width);
    const long long index =
        (((long long)batch_index * height + yy) * width + xx) * 3 + channel;
    return input[index];
}

__device__ __forceinline__ float bilinear_rgb(
    const float* input,
    int batch_index,
    float y,
    float x,
    int channel,
    int height,
    int width
) {
    const int x0 = (int)floorf(x);
    const int y0 = (int)floorf(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = x - (float)x0;
    const float wy = y - (float)y0;

    const float p00 = read_rgb(input, batch_index, y0, x0, channel, height, width);
    const float p10 = read_rgb(input, batch_index, y0, x1, channel, height, width);
    const float p01 = read_rgb(input, batch_index, y1, x0, channel, height, width);
    const float p11 = read_rgb(input, batch_index, y1, x1, channel, height, width);

    const float top = p00 + (p10 - p00) * wx;
    const float bottom = p01 + (p11 - p01) * wx;
    return top + (bottom - top) * wy;
}

__global__ void affine_color_kernel(
    const float* input,
    float* output,
    const int* horizontal_flip,
    const int* vertical_flip,
    const float* angles_degrees,
    const float* translations_x,
    const float* translations_y,
    const float* scales,
    const float* brightness,
    const float* contrast,
    int batch_size,
    int height,
    int width
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int batch_index = (int)blockIdx.z;
    if (x >= width || y >= height || batch_index >= batch_size) return;

    const float center_x = ((float)width - 1.0f) * 0.5f;
    const float center_y = ((float)height - 1.0f) * 0.5f;
    const float radians = angles_degrees[batch_index] * 0.01745329251994329577f;
    const float alpha = scales[batch_index] * cosf(radians);
    const float beta = scales[batch_index] * sinf(radians);

    const float matrix_02 =
        (1.0f - alpha) * center_x - beta * center_y + translations_x[batch_index];
    const float matrix_12 =
        beta * center_x + (1.0f - alpha) * center_y + translations_y[batch_index];

    const float shifted_x = (float)x - matrix_02;
    const float shifted_y = (float)y - matrix_12;
    const float determinant = alpha * alpha + beta * beta;

    float source_x = (alpha * shifted_x - beta * shifted_y) / determinant;
    float source_y = (beta * shifted_x + alpha * shifted_y) / determinant;

    if (horizontal_flip[batch_index]) source_x = ((float)width - 1.0f) - source_x;
    if (vertical_flip[batch_index]) source_y = ((float)height - 1.0f) - source_y;

    const long long output_base =
        (((long long)batch_index * height + y) * width + x) * 3;
    const float current_contrast = contrast[batch_index];
    const float current_brightness = brightness[batch_index];

    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        float value = bilinear_rgb(
            input,
            batch_index,
            source_y,
            source_x,
            channel,
            height,
            width
        );
        value = value * current_contrast + current_brightness;
        output[output_base + channel] = fminf(1.0f, fmaxf(0.0f, value));
    }
}

__global__ void gaussian_horizontal_kernel(
    const float* input,
    float* output,
    const float* weights,
    int radius,
    int batch_size,
    int height,
    int width
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int batch_index = (int)blockIdx.z;
    if (x >= width || y >= height || batch_index >= batch_size) return;

    const long long output_base =
        (((long long)batch_index * height + y) * width + x) * 3;
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            sum += weights[offset + radius] *
                read_rgb(input, batch_index, y, x + offset, channel, height, width);
        }
        output[output_base + channel] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const float* input,
    float* output,
    const float* weights,
    int radius,
    int batch_size,
    int height,
    int width
) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int batch_index = (int)blockIdx.z;
    if (x >= width || y >= height || batch_index >= batch_size) return;

    const long long output_base =
        (((long long)batch_index * height + y) * width + x) * 3;
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            sum += weights[offset + radius] *
                read_rgb(input, batch_index, y + offset, x, channel, height, width);
        }
        output[output_base + channel] = fminf(1.0f, fmaxf(0.0f, sum));
    }
}

}
'''


@dataclass(frozen=True)
class CuPyParameters:
    horizontal_flip: Any
    vertical_flip: Any
    angles_degrees: Any
    translations_x: Any
    translations_y: Any
    scales: Any
    brightness: Any
    contrast: Any
    blur_weights: Any | None
    blur_radius: int


@dataclass
class CuPyWorkspace:
    stage: Any
    blur_temporary: Any
    output: Any


@dataclass(frozen=True)
class CuPyLaunchGeometry:
    block_x: int
    block_y: int
    threads_per_block: int
    grid_x: int
    grid_y: int
    grid_z: int
    launched_threads: int


_AFFINE_KERNEL: Any | None = None
_BLUR_HORIZONTAL_KERNEL: Any | None = None
_BLUR_VERTICAL_KERNEL: Any | None = None


def cupy_imported() -> bool:
    return cp is not None


def cupy_cuda_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def cupy_version() -> str:
    return getattr(cp, "__version__", "not installed") if cp is not None else "not installed"


def cupy_device_name() -> str:
    if not cupy_cuda_available() or cp is None:
        return ""
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get("name", properties.get(b"name", b""))
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def cupy_compute_capability() -> str:
    if not cupy_cuda_available() or cp is None:
        return ""
    capability = cp.cuda.Device(0).compute_capability
    if isinstance(capability, bytes):
        capability = capability.decode()
    capability_text = str(capability)
    if len(capability_text) == 2 and capability_text.isdigit():
        return f"{capability_text[0]}.{capability_text[1]}"
    return capability_text


def _require_cupy() -> Any:
    if cp is None:
        raise RuntimeError(
            "CuPy is not installed. Install cupy-cuda13x to test explicit CUDA block sizes."
        )
    if not cupy_cuda_available():
        raise RuntimeError("CuPy does not detect an available CUDA device.")
    return cp


def _kernels() -> tuple[Any, Any, Any]:
    global _AFFINE_KERNEL, _BLUR_HORIZONTAL_KERNEL, _BLUR_VERTICAL_KERNEL
    module = _require_cupy()
    if _AFFINE_KERNEL is None:
        _AFFINE_KERNEL = module.RawKernel(AFFINE_COLOR_SOURCE, "affine_color_kernel")
        _BLUR_HORIZONTAL_KERNEL = module.RawKernel(
            AFFINE_COLOR_SOURCE, "gaussian_horizontal_kernel"
        )
        _BLUR_VERTICAL_KERNEL = module.RawKernel(
            AFFINE_COLOR_SOURCE, "gaussian_vertical_kernel"
        )
    return _AFFINE_KERNEL, _BLUR_HORIZONTAL_KERNEL, _BLUR_VERTICAL_KERNEL


def numpy_float_batch(images: list[np.ndarray]) -> np.ndarray:
    # Convert a uint8 RGB batch to float32 in the [0, 1] range.
    return np.stack(images, axis=0).astype(np.float32, copy=False) / 255.0


def host_float_to_cupy(batch: np.ndarray) -> Any:
    module = _require_cupy()
    return module.asarray(batch)


def numpy_batch_to_cupy(images: list[np.ndarray]) -> Any:
    return host_float_to_cupy(numpy_float_batch(images))


def cupy_to_host_float(array: Any) -> np.ndarray:
    module = _require_cupy()
    return module.asnumpy(array)


def host_float_to_numpy_batch(host: np.ndarray) -> list[np.ndarray]:
    converted = np.clip(np.rint(host * 255.0), 0.0, 255.0).astype(np.uint8)
    return [image.copy() for image in converted]


def cupy_to_numpy_batch(array: Any) -> list[np.ndarray]:
    return host_float_to_numpy_batch(cupy_to_host_float(array))


def prepare_cupy_parameters(
    profile: AugmentationProfile,
    batch_size: int,
    width: int,
    height: int,
) -> CuPyParameters:
    module = _require_cupy()
    values = [parameters_for_sample(profile, index) for index in range(batch_size)]

    blur_radius = profile.blur_kernel // 2 if profile.blur_kernel > 1 else 0
    blur_weights = None
    if blur_radius > 0:
        positions = np.arange(-blur_radius, blur_radius + 1, dtype=np.float32)
        sigma = max(float(profile.blur_sigma), 1e-6)
        weights = np.exp(-(positions * positions) / (2.0 * sigma * sigma)).astype(np.float32)
        weights /= weights.sum()
        blur_weights = module.asarray(weights)

    return CuPyParameters(
        horizontal_flip=module.asarray(
            [int(item.horizontal_flip) for item in values], dtype=module.int32
        ),
        vertical_flip=module.asarray(
            [int(item.vertical_flip) for item in values], dtype=module.int32
        ),
        angles_degrees=module.asarray(
            [item.angle_degrees for item in values], dtype=module.float32
        ),
        translations_x=module.asarray(
            [item.translate_x_fraction * width for item in values], dtype=module.float32
        ),
        translations_y=module.asarray(
            [item.translate_y_fraction * height for item in values], dtype=module.float32
        ),
        scales=module.asarray([item.scale for item in values], dtype=module.float32),
        brightness=module.asarray(
            [item.brightness_delta for item in values], dtype=module.float32
        ),
        contrast=module.asarray(
            [item.contrast_factor for item in values], dtype=module.float32
        ),
        blur_weights=blur_weights,
        blur_radius=blur_radius,
    )


def create_workspace(batch_size: int, height: int, width: int) -> CuPyWorkspace:
    # Allocate temporary device buffers reused across repeated kernel launches.
    module = _require_cupy()
    shape = (batch_size, height, width, 3)
    return CuPyWorkspace(
        stage=module.empty(shape, dtype=module.float32),
        blur_temporary=module.empty(shape, dtype=module.float32),
        output=module.empty(shape, dtype=module.float32),
    )


def launch_geometry(
    width: int,
    height: int,
    batch_size: int,
    block: tuple[int, int],
) -> CuPyLaunchGeometry:
    block_x, block_y = block
    if block_x <= 0 or block_y <= 0:
        raise ValueError("CUDA block dimensions must be positive.")
    threads = block_x * block_y
    if threads > 1024:
        raise ValueError(f"CUDA block {block_x}x{block_y} has more than 1024 threads.")
    grid_x = math.ceil(width / block_x)
    grid_y = math.ceil(height / block_y)
    grid_z = batch_size
    return CuPyLaunchGeometry(
        block_x=block_x,
        block_y=block_y,
        threads_per_block=threads,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        launched_threads=grid_x * grid_y * grid_z * threads,
    )


def apply_cupy_device(
    input_batch: Any,
    parameters: CuPyParameters,
    workspace: CuPyWorkspace,
    block: tuple[int, int],
) -> Any:
    module = _require_cupy()
    if input_batch.ndim != 4 or input_batch.shape[-1] != 3:
        raise ValueError("CuPy input must use shape [batch, height, width, 3].")

    batch_size, height, width, _ = input_batch.shape
    geometry = launch_geometry(width, height, batch_size, block)
    grid = (geometry.grid_x, geometry.grid_y, geometry.grid_z)
    cuda_block = (geometry.block_x, geometry.block_y, 1)
    affine_kernel, horizontal_kernel, vertical_kernel = _kernels()

    affine_kernel(
        grid,
        cuda_block,
        (
            input_batch,
            workspace.stage,
            parameters.horizontal_flip,
            parameters.vertical_flip,
            parameters.angles_degrees,
            parameters.translations_x,
            parameters.translations_y,
            parameters.scales,
            parameters.brightness,
            parameters.contrast,
            np.int32(batch_size),
            np.int32(height),
            np.int32(width),
        ),
    )

    if parameters.blur_radius <= 0 or parameters.blur_weights is None:
        return workspace.stage

    horizontal_kernel(
        grid,
        cuda_block,
        (
            workspace.stage,
            workspace.blur_temporary,
            parameters.blur_weights,
            np.int32(parameters.blur_radius),
            np.int32(batch_size),
            np.int32(height),
            np.int32(width),
        ),
    )
    vertical_kernel(
        grid,
        cuda_block,
        (
            workspace.blur_temporary,
            workspace.output,
            parameters.blur_weights,
            np.int32(parameters.blur_radius),
            np.int32(batch_size),
            np.int32(height),
            np.int32(width),
        ),
    )
    return workspace.output


def synchronize_cupy() -> None:
    module = _require_cupy()
    module.cuda.get_current_stream().synchronize()


def used_memory_mb() -> float:
    module = _require_cupy()
    return module.get_default_memory_pool().used_bytes() / (1024.0 ** 2)


def clear_cupy_memory() -> None:
    if cp is None:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
