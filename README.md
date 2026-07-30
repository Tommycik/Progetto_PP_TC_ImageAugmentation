# Parallel Image Augmentation with Albumentations and Kornia

## Project overview

This project benchmarks image augmentation on the CPU and GPU from a Python application executed directly in PyCharm. Albumentations provides the sequential and multithread CPU paths. Kornia and PyTorch provide a batch-oriented implementation on the CPU and CUDA.

The benchmark uses deterministic transformation parameters. Every backend receives the same profile, image order and per-sample values, so timing and output comparisons refer to the same logical workload.

No command-line arguments are required. Running `main.py` opens the interactive menu.

## Implementations

The benchmark evaluates five execution paths:

- **Albumentations sequential CPU** processes the batch with a normal loop.
- **Albumentations ThreadPool CPU** processes independent images with 1, 2, 4, 6, 8 and 12 workers.
- **Kornia CPU batch** applies the transformation chain to a BCHW PyTorch tensor.
- **Kornia CUDA end-to-end** includes transfer to the GPU, augmentation and transfer back to the CPU.
- **Kornia CUDA device-only** measures augmentation after the input tensor is already stored on the GPU.

OpenCV internal threads are fixed to one. PyTorch CPU intra-operation and inter-operation threads are also fixed to one. This keeps the explicit worker counts controlled.

## Augmentation profiles

The project contains seven deterministic profiles:

- **Identity** applies no transformation and measures framework overhead.
- **GeometricLight** applies a horizontal flip and mild affine motion.
- **GeometricStrong** applies horizontal and vertical flips with stronger affine motion.
- **Color** changes brightness and contrast.
- **Blur** applies Gaussian blur.
- **MixedLight** combines light geometric, colour and blur operations.
- **MixedStrong** combines stronger geometric, colour and blur operations.

The transformation direction and magnitude change with the sample index using a fixed cycle. Repeated executions therefore use the same parameter sequence.

## Full benchmark

The benchmark covers:

- `512x512` with batches `1, 4, 8, 16, 32, 64`;
- `1024x1024` with batches `1, 4, 8, 16, 32`;
- `2048x2048` with batches `1, 4, 8`;
- CPU worker counts `1, 2, 4, 6, 8, 12`;
- two warm-up executions;
- five measured repetitions.

Seven profiles and fourteen resolution-batch combinations produce **98 workloads**. Every workload writes ten backend rows, for a total of **980 CSV rows**.

For every workload the program measures the sequential path and all ThreadPool sizes. The fastest exact Albumentations result becomes the CPU reference for the Kornia comparison.

## Timing data

The CSV records:

- mean, median, minimum and maximum time;
- population standard deviation;
- coefficient of variation;
- sequential baseline time;
- fastest CPU time;
- speedup against the sequential path;
- speedup against the fastest CPU path;
- CPU parallel efficiency;
- images per second;
- megapixels per second;
- CUDA peak allocated memory;
- timing scope, status and notes.

## Output verification

The benchmark records:

- exact equality;
- tolerance equality;
- different values and pixels;
- MAE;
- RMSE;
- maximum difference;
- PSNR;
- global SSIM.

Albumentations ThreadPool is compared with the sequential Albumentations result. Kornia CPU and CUDA are compared with the corresponding CPU references. Small cross-library differences can occur because interpolation, border handling and floating-point rounding are implemented independently.

## Automatic preview

The preview uses fixed parameters:

```text
Profile: MixedStrong
Resolution: 512x512
Batch size: 8
```

The program measures the available complete paths:

- Albumentations ThreadPool;
- Kornia CPU;
- Kornia CUDA end-to-end when CUDA is available.

The backend with the lowest measured mean time generates the saved preview. The output directory contains the original images, augmented images and `contact_sheet.png`.

## Input formats

The program accepts one image or a directory containing images with these extensions:

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

Pressing Enter without a path uses a deterministic synthetic image. When the batch is larger than the number of source images, the loaded images are repeated.

## Project structure

```text
main.py
Src/
├── __init__.py
├── app.py
├── config.py
├── backends.py
└── benchmark.py
requirements.txt
Output/
├── Results/
└── Previews/
```

- `main.py` starts the application.
- `Src/app.py` contains the menu, image loading, batch construction, automatic preview and environment report.
- `Src/config.py` contains paths, profiles, deterministic parameters and the benchmark plan.
- `Src/backends.py` contains the Albumentations and Kornia implementations.
- `Src/benchmark.py` contains timing, metrics, backend execution and CSV generation.
- `Src/__init__.py` marks `Src` as a Python package.

## Requirements

The project requires:

- Python;
- NumPy;
- Pillow;
- OpenCV;
- Albumentations;
- PyTorch;
- Kornia;
- an NVIDIA GPU and a compatible CUDA-enabled PyTorch installation for GPU execution.

## Use from PyCharm

1. Open the project directory in PyCharm.
2. Select the project virtual environment.
3. Install the packages from `requirements.txt` and the correct PyTorch build.
4. Open `main.py`.
5. Press the normal PyCharm Run button.

The menu is:

```text
1. Full benchmark
2. Generate automatic preview
3. Check environment
4. Exit
```

The benchmark starts after the input path is selected. The preview automatically selects its backend and parameters.

## Output directories

Benchmark CSV files are written to:

```text
Output/Results
```

Preview images are written to:

```text
Output/Previews
```
