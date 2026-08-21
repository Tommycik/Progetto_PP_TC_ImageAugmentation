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

The CSV records mean, median, minimum, maximum, population standard deviation and coefficient of variation.

- **TimeMean_ms** is the arithmetic mean of the measured repetitions and is the main execution-time value used for performance comparisons.
- **TimeMedian_ms** is the middle measured time after sorting the repetitions. It is less sensitive than the mean to isolated slow or fast samples.
- **TimeMin_ms** and **TimeMax_ms** are the fastest and slowest measured repetitions and show the observed timing range.
- **StdDev_ms** is the population standard deviation in milliseconds and measures the absolute dispersion of repeated measurements.
- **CoefficientVariation_percent** expresses timing variation relative to the mean:

```text
CV = standard deviation / mean x 100
```

A lower CV indicates more stable repeated measurements and makes timing variability comparable across workloads with very different execution times.

The CSV also stores the baselines and speedups used for the different comparisons:

- **SequentialBaseline_ms** is the Albumentations sequential time for the same workload.
- **BestCpu_ms** is the fastest valid CPU time selected across Albumentations sequential, Albumentations ThreadPool and Kornia CPU.
- **Speedup_vs_Sequential** is the sequential baseline divided by the measured time.
- **Speedup_vs_BestCPU** is the fastest valid CPU time divided by the measured time and is the direct comparison used for CUDA.
- **ParallelBaseline_ms** is the baseline used for internal CPU scaling. Albumentations uses its sequential result while Kornia CPU uses its one-thread tensor result.
- **ParallelSpeedup** is:

```text
parallel speedup = parallel baseline / parallel time
```

- **ParallelEfficiency** is:

```text
parallel efficiency = parallel speedup / number of workers or threads
```

Parallel efficiency is used only when an explicit CPU worker or thread count exists. It is not used as a CUDA efficiency measure.

The CSV records two throughput metrics:

- **Throughput_images_s** is the number of complete images processed per second. It shows how effectively batching amortizes scheduling, framework and launch overhead.
- **Throughput_MPixels_s** is the number of millions of image pixels processed per second. It also includes image resolution and is useful when comparing workloads with different image sizes.

They are calculated as:

```text
Throughput_images_s = BatchSize x 1000 / TimeMean_ms
Throughput_MPixels_s = BatchSize x Width x Height / TimeMean_ms / 1000
```

**CudaPeakMemory_MB** is the peak CUDA memory allocated during the GPU execution. It is useful for relating batch limits to image resolution and to the memory required by input, output and intermediate tensors.

## Output verification

The benchmark records:

- **ExactMatch**, which is true only when every corresponding RGB channel value is identical;
- **ToleranceMatch**, which indicates whether the output satisfies the configured numerical acceptance rule even when it is not exactly identical;
- **DifferentValues**, which counts individual RGB channel values that differ;
- **DifferentPixels**, which counts pixels for which at least one RGB channel differs;
- **MAE**, the mean absolute error between corresponding 8-bit channel values. Zero means identical values and lower values indicate a smaller average difference;
- **RMSE**, the root mean square error. Larger isolated errors have more influence than in MAE because the differences are squared before averaging;
- **MaxDifference**, the largest absolute difference found in any corresponding channel value. It helps detect a local large error that could be hidden by a small average error;
- **PSNR_dB**, the peak signal-to-noise ratio derived from RMSE on the 0-255 image scale. Higher values indicate closer images and an exact match produces infinite PSNR;
- **GlobalSSIM**, which measures global structural similarity. A value of one indicates identical structure. The project calculates one value for each complete RGB channel and averages the three results.

Tolerance matching is useful because Albumentations, Kornia CPU and Kornia CUDA can use different interpolation, floating-point arithmetic and rounding while still applying the same intended transformation.

### Metrics used in the final report and presentation

The CSV intentionally contains more metrics than the final report and presentation. The final documents use the mean execution time, minimum and maximum values where relevant, standard deviation and CV, internal CPU speedup and parallel efficiency, CUDA speedup against the fastest valid CPU result, CUDA peak memory, exact or tolerance verification, MAE and global SSIM. The report discusses throughput mainly through time per image instead of presenting the raw throughput columns directly.

`TimeMedian_ms`, `DifferentValues`, `DifferentPixels`, `RMSE`, `PSNR_dB`, `Throughput_images_s` and `Throughput_MPixels_s` are retained mainly for diagnostics, reproducibility and deeper analysis but are not presented directly as result metrics in the final report or presentation. `MaxDifference` is also not presented directly, although it contributes to the tolerance rule used to validate outputs.

## Results interpretation

Albumentations is the fastest CPU library in all 112 workloads. Its ThreadPool implementation is the fastest CPU configuration in 87 workloads while the sequential path remains best in 25 smaller or lighter cases. The maximum internal CPU speedup is `5.47x` for Albumentations and `3.03x` for Kornia CPU. The two libraries scale differently because Albumentations distributes independent images among workers while Kornia uses PyTorch intra-operation parallelism inside tensor operations.

Kornia CUDA device-only is faster than the best CPU result in 61 of the 112 workloads and reaches a maximum speedup of `10.76x`. Complete CUDA end-to-end execution is faster in 12 workloads and reaches `2.93x`. Small images and light profiles are dominated by framework, kernel-launch and transfer overhead, while larger resolutions and more expensive geometric or mixed profiles provide enough work to make GPU execution advantageous.

Device-only and end-to-end timings describe different use cases. Device-only isolates the augmentation kernels after the tensors are already on the GPU, while end-to-end also includes host-to-device transfer, device-to-host transfer and final synchronization. In the largest MixedStrong workloads the device interval represents only about 38 percent of the end-to-end time, showing that transfers and host-side overhead remain significant when the augmentation is used as an isolated stage connected to CPU memory.

All 1680 result rows satisfy the configured tolerance. Albumentations ThreadPool is exactly equal to its sequential reference in every workload. Kornia CPU and CUDA end-to-end have a mean MAE close to `0.42` against the Albumentations reference and their minimum global SSIM remains above `0.996`. CUDA device-only compared with Kornia CPU one-thread has a mean MAE of `0.000106` and a minimum SSIM of `0.999999`. The larger output differences therefore come mainly from the different Albumentations and Kornia implementations rather than from moving the same Kornia pipeline from CPU to CUDA.


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
