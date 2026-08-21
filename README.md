# Parallel Image Augmentation with Albumentations and Kornia

## Project overview

This project benchmarks deterministic image augmentations on the CPU and GPU from a Python application. Albumentations provides an image-oriented CPU implementation while Kornia provides one batch-oriented PyTorch implementation that can execute on the CPU or CUDA.

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

The project evaluates the same Kornia pipeline under different execution conditions.

### Kornia CPU

The same CPU tensor operation is measured with these PyTorch intra-operation thread counts:

```text
1, 2, 4, 6, 8, 12
```

PyTorch inter-operation parallelism remains fixed to one.

Tensor conversion is outside the full benchmark timing. The one-thread Kornia result is the scaling baseline for the other Kornia CPU tests.

### Kornia CUDA

The Kornia pipeline is executed on CUDA tensors and runs on the GPU.

The same CUDA execution is recorded with two timing scopes:

- **End-to-end scope** includes the CPU-to-GPU tensor copy, Kornia augmentation on the GPU and the GPU-to-CPU result copy.
- **Device-only scope** measures only the Kornia augmentation after the input and parameter tensors are already on the GPU.

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
- `4096x4096` with batches `1, 2`;
- CPU thread counts `1, 2, 4, 6, 8, 12`;
- two warm-up executions;
- five measured repetitions.

The maximum batch at `4096x4096` is two because memory usage grows with both image area and the number of tensors retained by the Kornia pipeline. One RGB `float32` tensor at this resolution requires approximately 192 MiB per image, so batch two requires about 384 MiB for one tensor alone. Input, output, affine, blur and transfer tensors coexist during execution. MixedStrong at batch two reaches a measured CUDA peak of approximately 2312.9 MB. Larger batches were excluded to reduce GPU and system-memory pressure and avoid out-of-memory failures.

Seven profiles and sixteen resolution-batch combinations produce 112 workloads. Every workload executes:

```text
1 Albumentations sequential test
6 Albumentations ThreadPool tests
6 Kornia CPU thread tests
1 Kornia CUDA execution recorded with 2 timing scopes
```

The CUDA execution produces one end-to-end row and one device-only row. Every workload therefore produces 15 measured result rows:

```text
112 x 15 = 1680 result rows
```

The fastest valid CPU configuration is selected across Albumentations and Kornia for each workload.

## Timing metrics

The CSV stores the execution time of every configuration using several simple statistics:

- **TimeMean_ms** is the average execution time and is the main value used to compare performance.
- **TimeMedian_ms** is the middle measured time and is useful when one repetition is unusually slow or fast.
- **TimeMin_ms** and **TimeMax_ms** are the fastest and slowest repetitions.
- **StdDev_ms** shows how much the measured times vary around the mean.
- **CoefficientVariation_percent** expresses the same variability as a percentage of the mean:

```text
CV = standard deviation / mean x 100
```

A low CV means that repeated measurements are stable.

The CSV also stores the reference times used to calculate speedups:

- **SequentialBaseline_ms** is the Albumentations sequential time for the same workload.
- **BestCpu_ms** is the fastest valid CPU time for that workload.
- **Speedup_vs_Sequential** shows how many times faster a configuration is than the sequential baseline.
- **Speedup_vs_BestCPU** compares a configuration with the fastest CPU result. This is the main speedup used for CUDA comparisons.
- **ParallelBaseline_ms** is the reference used to study CPU scaling. Albumentations uses its sequential execution, while Kornia CPU uses its one-thread execution.
- **ParallelSpeedup** shows how much faster the parallel execution is than its own baseline.
- **ParallelEfficiency** relates that speedup to the number of workers or threads:

```text
parallel efficiency = parallel speedup / number of workers or threads
```

An efficiency close to one means that the added CPU workers are being used very effectively. This metric is only used for CPU configurations, not for CUDA.

Two throughput values are also saved:

- **Throughput_images_s** is the number of complete images processed each second.
- **Throughput_MPixels_s** is the number of millions of pixels processed each second.

These values are useful for understanding how much work a configuration can process, especially when batch size or image resolution changes.

```text
Throughput_images_s = BatchSize x 1000 / TimeMean_ms
Throughput_MPixels_s = BatchSize x Width x Height / TimeMean_ms / 1000
```

**CudaPeakMemory_MB** is the maximum CUDA memory used during the GPU execution. It helps explain why larger images require smaller maximum batch sizes.

## Output verification

Performance is useful only if the different implementations still produce equivalent outputs. For this reason, the CSV stores several comparison metrics:

- **ExactMatch** checks whether every output value is exactly equal to the reference.
- **ToleranceMatch** checks whether small numerical differences are still inside the accepted limits.
- **DifferentValues** counts how many individual RGB values are different.
- **DifferentPixels** counts how many pixels contain at least one different channel.
- **MAE** measures the average absolute difference between corresponding values. Lower is better and zero means an exact match.
- **RMSE** is similar to MAE but gives more importance to larger differences.
- **MaxDifference** is the largest single difference found between the two outputs.
- **PSNR_dB** is another image-error measure derived from RMSE. Higher values mean that the two images are more similar.
- **GlobalSSIM** measures overall structural similarity. Values close to one indicate very similar images.

These metrics are useful because Albumentations and Kornia can produce slightly different numerical values even when they apply the same intended augmentation.

### Metrics used in the final report and presentation

The CSV contains more information than was needed in the final report and presentation.

The final documents mainly use mean execution time, standard deviation, CV, speedup, CPU parallel efficiency, CUDA peak memory, exact or tolerance matching, MAE and global SSIM. Minimum and maximum times are also used when discussing measurement stability.

`TimeMedian_ms`, `DifferentValues`, `DifferentPixels`, `RMSE`, `PSNR_dB`, `Throughput_images_s` and `Throughput_MPixels_s` are stored mainly for additional inspection and are not presented directly as result metrics in the final report or presentation. `MaxDifference` is also not shown directly, but it is used by the tolerance check.

## Results interpretation

The results show that Albumentations is the strongest CPU option for this benchmark, its ThreadPool version benefits from processing independent images in parallel. Kornia CPU also gains from additional threads but its scaling is more limited.

CUDA becomes useful only when the workload is large enough. The device-only timing shows the potential performance when the data is already on the GPU while the end-to-end timing is more representative when images start and end in the CPU memory because it also includes data transfers.

The output comparisons show that all configurations remain within the accepted tolerance. Albumentations ThreadPool matches its sequential reference exactly while Kornia CPU and CUDA show only small numerical differences. Most of the larger differences come from comparing the Albumentations and Kornia implementations rather than from changing Kornia from CPU to CUDA.


## Preview generation

The preview uses one fixed configuration:

```text
Backend: Albumentations ThreadPool
CPU workers: 8
Profile: MixedStrong
Resolution: 512x512
Batch size: 8
```

The preview mode applies the prepared Albumentations transformations with eight workers, saves the input and output images and creates a contact sheet.

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
├── metrics.py
├── report.py
└── benchmark.py
requirements.txt
Output/
├── Results/
└── Previews/
```

- `main.py` starts the application.
- `Src/app.py` contains the menu, image loading, batch creation, fixed preview and environment report.
- `Src/config.py` contains paths, profiles, deterministic parameters and the benchmark plan.
- `Src/backends.py` contains the unchanged Albumentations and Kornia operations.
- `Src/metrics.py` contains the timing and output-comparison functions.
- `Src/report.py` contains the CSV schema and row helper functions.
- `Src/benchmark.py` retains workload iteration, backend orchestration, CSV opening and row-writing calls.
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
