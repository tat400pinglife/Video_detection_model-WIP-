# Video Processing Optimization Report
**Module:** `gpu_proccesor.py`
**Objective:** Accelerate the baseline data processing pipeline by migrating sequential CPU operations to parallel CUDA hardware architectures.

## 1. Executive Summary
The baseline feature extraction pipeline (`space.py`) relies heavily on CPU-bound libraries (`numpy`, `cv2`) to compute complex forensic features. While mathematically accurate, executing sequential `for` loops on high-resolution video frames creates a significant bottleneck, especially when processing large datasets scraped from the web.

The `gpu_proccesor.py` module was engineered to solve this by pushing raw video frames directly into Video RAM (VRAM) and utilizing PyTorch tensor operations. This allows the GPU's CUDA cores to process all 32 frames of a video clip simultaneously.

## 2. Key Algorithmic Upgrades

### A. Temporal Motion (Vector Subtraction)
* **Baseline (`space.py`):** Calculates motion by iterating through frames with a `for` loop, applying `np.abs(gray[i] - gray[i-1])` sequentially.
* **CUDA Optimization:** Replaced the loop with instantaneous tensor slicing: `torch.abs(gray[1:] - gray[:-1])`. The GPU subtracts the shifted sequence from the original sequence in a single parallel operation.

### B. PRNU Noise Fingerprint (Convolution vs. Looping)
* **Baseline (`space.py`):** Isolates camera noise by applying `cv2.GaussianBlur` inside a loop for 5 separate frames, forcing the CPU to stop and restart the math 5 times.
* **CUDA Optimization:** Engineered a custom 5x5 Gaussian Kernel tensor natively on the GPU. By using PyTorch's `F.conv2d`, the blur matrix is applied to the entire 5-frame volume simultaneously, eliminating the loop entirely.

### C. Frequency Analysis (Hardware FFT)
* **Baseline (`space.py`):** Uses standard `np.fft.fft2`, which relies on the CPU to calculate complex trigonometric Fourier transforms.
* **CUDA Optimization:** Transitioned to `torch.fft.fft2`, routing the calculations directly through NVIDIA's specialized hardware-accelerated FFT architecture.

## 3. Data Compression & Memory Alignment
To ensure compatibility with the rest of the project pipeline, the GPU processor perfectly replicates the memory-saving techniques established in the baseline script.

By default, PyTorch computes all GPU math using 32-bit floats. Before transferring the data back to the CPU for SSD storage, `gpu_proccesor.py` mimics the exact `compress_features` logic found in `space.py`:
* **Spatial Data (`rgb_batch`):** Clamped to `0-255` and downcast to `torch.uint8` (8-bit integers).
* **Mathematical Maps (`diff_seq`, `fft`, `prnu`):** Downcast to `torch.float16` (Half-precision floats).

This final step ensures that the resulting `.pt` tensors are identical in size and shape to the baseline, saving hundreds of gigabytes of disk space while processing at a fraction of the time.