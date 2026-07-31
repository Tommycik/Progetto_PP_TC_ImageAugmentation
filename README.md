# Parallel Image Augmentation with Albumentations and Kornia

## Project overview

This project benchmarks deterministic image augmentation on the CPU and GPU from a Python application executed directly in PyCharm. Albumentations provides an image-oriented CPU implementation. Kornia provides one batch-oriented PyTorch implementation that can execute on the CPU or CUDA.

Every implementation receives the same images, profiles and deterministic per-sample parameters. The benchmark records timing, scalability and output-comparison data in CSV format.

Running `main.py` opens the interactive menu. No command-line arguments are required.

## Images and batches

A batch is a group of images processed in one benchmark call. Its size is identified by `BatchSize`.

Albumentations receives the batch as a Python list of RGB NumPy images. Independent images can be assigned to different `ThreadPoolExecutor` workers.

Kornia converts the same batch to one PyTorch tensor with shape:

```text
B x C x H x W
```

`B` is the number of images, `C` is three for RGB data and `H` and `W` are the image dimensions.

## Albumentations CPU

### Sequential

The sequential implementation applies one prepared transformation after another with a normal Python loop. OpenCV internal threads are fixed to one so this path does not use hidden CPU parallelism.

### ThreadPool

The parallel implementation assigns independent images to a reused `ThreadPoolExecutor`. The tested worker counts are:

```text
1, 2, 4, 6, 8, 12
```

Thread-pool construction is outside the timed region. The output must be exactly equal to the sequential Albumentations result.

## Kornia pipeline

Kornia uses one transformation function for all measurements. The function applies flips, affine transformation, brightness, contrast and Gaussian blur to a complete tensor batch.

The project does not contain three different Kornia algorithms. It evaluates the same Kornia pipeline under different execution conditions.

### Kornia CPU

The same CPU tensor operation is measured with these PyTorch intra-operation thread counts:

```text
1, 2, 4, 6, 8, 12
```

PyTorch inter-operation parallelism remains fixed to one. `torch.utils.benchmark.Timer` applies the selected intra-operation thread-pool size during each measurement.

Tensor conversion is outside the full benchmark timing. The one-thread Kornia result is the scaling baseline for the other Kornia CPU rows.

### Kornia CUDA

The Kornia pipeline is executed on CUDA tensors. The augmentation runs on the GPU.

The same CUDA execution is recorded with two timing scopes:

- **End-to-end scope** includes the CPU-to-GPU tensor copy, Kornia augmentation on the GPU and the GPU-to-CPU result copy.
- **Device-only scope** measures only the Kornia augmentation after the input and parameter tensors are already on the GPU.

These rows are two timings of one CUDA pipeline. They are not separate augmentation implementations.

The device-only value is useful when the previous and following stages also operate on GPU tensors. The end-to-end value is the direct comparison for a program whose input and output remain in CPU memory.

## Augmentation profiles

The benchmark contains seven deterministic profiles:

- **Identity** applies no transformation and measures framework overhead.
- **GeometricLight** applies a horizontal flip and mild affine motion.
- **GeometricStrong** applies horizontal and vertical flips with stronger affine motion.
- **Color** changes brightness and contrast.
- **Blur** applies Gaussian blur.
- **MixedLight** combines light geometric, colour and blur operations.
- **MixedStrong** combines stronger geometric, colour and blur operations.

The sign and magnitude of the transformation change with the sample index using a fixed cycle. Repeated executions therefore use the same logical workload.

## Full benchmark

The workload matrix contains:

- `512x512` with batches `1, 4, 8, 16, 32, 64`;
- `1024x1024` with batches `1, 4, 8, 16, 32`;
- `2048x2048` with batches `1, 4, 8`;
- CPU thread counts `1, 2, 4, 6, 8, 12`;
- two warm-up executions;
- five measured repetitions.

Seven profiles and fourteen resolution-batch combinations produce 98 workloads. Every workload writes:

```text
1 Albumentations sequential row
6 Albumentations ThreadPool rows
6 Kornia CPU thread rows
2 Kornia CUDA timing-scope rows
```

The complete benchmark therefore produces:

```text
98 x 15 = 1470 rows
```

The fastest valid CPU configuration is selected across Albumentations and Kornia for each workload.

## Timing metrics

The CSV records mean, median, minimum, maximum, population standard deviation and coefficient of variation.

The coefficient of variation is:

```text
CV = standard deviation / mean x 100
```

A lower CV indicates more stable repeated measurements.

Parallel speedup is calculated against the one-thread result of the same library:

```text
speedup = one-thread time / parallel time
```

Parallel efficiency is:

```text
efficiency = speedup / number of workers or threads
```

Albumentations uses its sequential time as the parallel baseline. Kornia CPU uses its one-thread tensor time.

The CSV also records throughput, speedup against the fastest valid CPU result and CUDA peak allocated memory.

## Output verification

The benchmark records:

- exact equality;
- tolerance equality;
- different values and pixels;
- mean absolute error;
- root mean square error;
- maximum difference;
- PSNR;
- global SSIM.

MAE is the average absolute difference between corresponding 8-bit channel values. Zero means identical values.

SSIM measures structural similarity. A value of one indicates identical images. The project calculates one global value for every RGB channel and averages the three results.

Albumentations ThreadPool is compared with Albumentations sequential. Kornia CPU and CUDA are compared using the same deterministic workload and tolerance thresholds.

## Preview generation

The preview uses one fixed configuration:

```text
Backend: Albumentations ThreadPool
CPU workers: 8
Profile: MixedStrong
Resolution: 512x512
Batch size: 8
```

Preview mode does not benchmark or compare backends. It directly applies the prepared Albumentations transformations with eight workers, saves the input and output images, and creates a contact sheet.

## Input formats

The program accepts one image or a directory containing:

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

Pressing Enter without a path uses a deterministic synthetic image. Source images are repeated when the requested batch is larger than the number of loaded images.

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
- `Src/app.py` contains the menu, image loading, batch creation, fixed preview and environment report.
- `Src/config.py` contains paths, profiles, deterministic parameters and the benchmark plan.
- `Src/backends.py` contains the Albumentations and Kornia operations.
- `Src/benchmark.py` contains timing, CPU thread scaling, CUDA timing scopes, metrics and CSV generation.
- `Src/__init__.py` marks `Src` as a Python package.

## Use from PyCharm

1. Open the project directory in PyCharm.
2. Select the project virtual environment.
3. Install the packages from `requirements.txt` and the appropriate PyTorch build.
4. Open `main.py`.
5. Use the normal PyCharm Run button.

The menu is:

```text
1. Full benchmark
2. Generate preview
3. Check environment
4. Exit
```

CSV files are written to `Output/Results`. Preview images and contact sheets are written to `Output/Previews`.
