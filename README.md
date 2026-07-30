# Image Augmentation Benchmark with Albumentations, Kornia and CuPy

## Project overview

This project compares sequential CPU, multithread CPU and GPU image augmentation from a Python application executed directly in PyCharm. The program uses Albumentations for the CPU image pipelines. It uses Kornia and PyTorch for batch-oriented execution on CPU and CUDA. It also uses CuPy RawKernel to test explicit CUDA block dimensions.

The benchmark is designed for a parallel-computing project. It keeps deterministic augmentation parameters so every backend receives the same logical workload. It records execution time, variability, throughput, speedup and output-comparison metrics in a CSV file.

The project does not require command-line arguments. `main.py` opens an interactive menu in the PyCharm console.

## Why three libraries are used

Albumentations is used because it provides a practical CPU image-augmentation pipeline over NumPy images. A plain loop forms the sequential reference. A `ThreadPoolExecutor` applies the same prepared transformations concurrently to different images.

Kornia is used because it applies image operations to PyTorch tensors. The same batch-oriented implementation can run on CPU or CUDA. This makes it possible to compare device behaviour while preserving the same Kornia processing path.

CuPy is used because Kornia and PyTorch do not expose the CUDA block dimensions selected internally by their kernels. CuPy RawKernel allows the project to launch custom CUDA kernels with explicit grid and block shapes. It therefore adds a low-level GPU experiment while the rest of the project remains Python-based.

## Augmentation profiles

The benchmark contains these profiles:

- **Identity** applies no transformation and measures framework, batching and transfer overhead.
- **GeometricLight** applies a horizontal flip and mild affine motion.
- **GeometricStrong** applies horizontal and vertical flips with stronger rotation, translation and scale changes.
- **Color** changes brightness and contrast.
- **Blur** applies a fixed Gaussian blur.
- **MixedLight** combines light geometric, colour and blur operations.
- **MixedStrong** combines stronger geometric, colour and blur operations.

The profiles use operations implemented by Albumentations, Kornia and CuPy. This keeps the comparison focused on execution strategy instead of giving one backend an operation that the others do not execute.

Each image in a batch receives deterministic parameters. Direction and magnitude change with the image index. Repeated benchmark runs therefore process the same parameter sequence.

## Implementations

### Albumentations sequential

This is the main CPU baseline. The program processes every image in the batch with a plain Python loop. OpenCV internal threads are limited to one so the baseline does not use hidden CPU parallelism.

### Albumentations threaded

The same prepared Albumentations transformations are distributed with a reused `ThreadPoolExecutor`. The benchmark evaluates 1, 2, 4, 6, 8 and 12 worker threads. Pool creation is excluded from the measured region.

### Kornia CPU

The input batch is converted to a BCHW floating-point PyTorch tensor. Kornia then applies flips, affine transformation, brightness, contrast and Gaussian blur to the full batch on the CPU.

### Kornia CUDA

Two CUDA timing scopes are recorded.

- **End-to-end** includes host-to-device transfer, augmentation and device-to-host transfer.
- **Device-only** keeps the input on the GPU and measures the augmentation stage.

The end-to-end result can be compared directly with the CPU execution. The device-only result describes the behaviour of a pipeline where the data already resides on the GPU.

### CuPy RawKernel CUDA

CuPy executes custom CUDA kernels with explicit block dimensions. The end-to-end path is measured with a fixed reference block. The device-only tests then compare all configured block layouts without repeating transfer overhead for every layout.

The benchmark evaluates:

```text
8x8
16x16
32x8
8x32
32x16
16x32
32x32
256x1
1x256
```

The batch index is mapped to the third grid dimension. The first two dimensions cover the image width and height.

## Full benchmark

The project contains one full benchmark. The smaller quick benchmark was removed because the university evaluation requires a complete and consistent dataset.

The workloads are:

- 512 by 512 with batch sizes 1, 4, 8, 16, 32 and 64;
- 1024 by 1024 with batch sizes 1, 4, 8, 16 and 32;
- 2048 by 2048 with batch sizes 1, 4 and 8.

Every measured configuration uses two warm-up executions and five measured repetitions. Larger batches are included because Kornia and CUDA are batch-oriented. The largest values are reduced at high resolution to limit host and GPU memory pressure.

## Timing and CSV metrics

The benchmark records:

- mean time;
- median time;
- minimum time;
- maximum time;
- population standard deviation;
- coefficient of variation;
- sequential baseline time;
- speedup;
- parallel efficiency for CPU worker configurations;
- images per second;
- megapixels per second;
- timer scope;
- warm-up count;
- repetition count;
- CUDA peak memory;
- CUDA block and grid dimensions.

The output comparison records:

- exact match;
- tolerance match;
- different values;
- different pixels;
- mean absolute error;
- root mean square error;
- maximum difference;
- PSNR;
- global SSIM.

Albumentations sequential and threaded are expected to match because they use the same prepared transformations. Cross-library comparisons may not be byte-identical because interpolation, border handling and numerical rounding can differ. MAE, PSNR and SSIM provide a more useful interpretation of these cases.

CSV files are created in:

```text
Output/Results
```

The filename contains the benchmark type and timestamp. Each row is flushed immediately. Completed rows remain available if a long benchmark is interrupted.

## Preview generation

Preview files are created in:

```text
Output/Previews
```

Every execution creates a timestamped subdirectory containing the input batch, the generated output folders and a `contact_sheet.png` file.

The preview menu provides three modes.

### Fast preview

This mode executes one backend only. It prefers Kornia CUDA when PyTorch detects the GPU. It then tries CuPy CUDA. If GPU support is unavailable it uses threaded Albumentations. No sequential reference or comparison metrics are calculated.

### Comparison preview

This mode executes Albumentations sequential, Albumentations threaded, Kornia CPU and the available CUDA backends. It saves every output and prints the main comparison metrics.

### Specific backend preview

This mode runs only the selected implementation. It is useful for inspecting one backend without executing the complete comparison.

## Input formats

The application accepts:

```text
.png
.jpg
.jpeg
.bmp
.tif
.tiff
.webp
.ppm
.pgm
```

The user may enter one image path or a directory containing multiple images. Pressing Enter without a path creates a deterministic synthetic image. When the requested batch is larger than the number of loaded images, the source images are repeated.

## Project structure

- `main.py` is the entry point used by PyCharm.
- `Src/app.py` contains the menu and environment report.
- `Src/config.py` contains output paths, profiles and the benchmark plan.
- `Src/profiles.py` generates deterministic parameters for each batch element.
- `Src/albumentations_backend.py` contains sequential and threaded CPU execution.
- `Src/kornia_backend.py` contains the Kornia CPU and CUDA pipeline.
- `Src/cupy_backend.py` contains the RawKernel CUDA implementation and block-size logic.
- `Src/benchmark.py` executes the benchmark and writes the CSV rows.
- `Src/preview.py` creates fast, comparative and backend-specific previews.
- `Src/metrics.py` calculates numerical comparison metrics.
- `Src/timing.py` calculates timing statistics.
- `Src/image_io.py` loads images, builds batches and writes preview files.
- `requirements.txt` lists the common Python dependencies.

The source remains divided into modules because the three backends have different responsibilities. Joining every file would make the CUDA kernels, benchmark logic and preview logic more difficult to inspect and maintain.

## Requirements

The project requires:

- Python installed and selected by PyCharm;
- NumPy;
- Pillow;
- OpenCV;
- Albumentations;
- PyTorch;
- Kornia;
- CuPy for explicit CUDA block tests;
- an NVIDIA GPU and compatible driver for CUDA execution.

The supplied installation command file contains the commands for a Windows virtual environment. The PyTorch and CuPy packages must match a CUDA runtime supported by the installed driver.

## Use from PyCharm

1. Open the project directory in PyCharm.
2. Select the project virtual environment as the Python interpreter.
3. Install the required packages using the provided installation command file.
4. Open `main.py`.
5. Run `main.py` with the normal PyCharm Run button.

No program arguments are required. The menu is:

```text
1. Full benchmark
2. Generate previews
3. Check environment
4. Exit
```

Use `Check environment` before the benchmark. It prints package versions, CPU information, CUDA availability, GPU name, compute capability and output directories.

## Output directories

```text
Output/
├── Results/
└── Previews/
```

`Results` contains CSV benchmark data. `Previews` contains timestamped image groups and contact sheets. Both directories are created automatically when the application starts.

## Result interpretation

CPU thread scaling depends on batch size. A batch of one image cannot provide useful image-level parallelism to many worker threads. Larger batches provide more independent tasks but also increase memory use.

Kornia device-only results should not be compared directly with CPU end-to-end results without stating the different timing scope. CuPy block-size rows are intended to compare CUDA layouts with one another. The end-to-end CuPy row includes data transfer and is the appropriate row for a direct CPU comparison.

The Identity profile is useful for measuring framework overhead. The geometric profiles emphasize interpolation. The colour profile contains mainly element-wise operations. Blur increases neighbourhood work and memory traffic. Mixed profiles represent a more complete augmentation pipeline.
