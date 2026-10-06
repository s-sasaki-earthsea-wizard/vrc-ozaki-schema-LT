# CUDA 12 vs CUDA 13.4 (Ozaki scheme) FP64 benchmark report

- conditions: cuda12, cuda13-native, cuda13-emu, cuda13-emu-eager
- baseline for speedup: `cuda12`; emulation effect isolated against `cuda13-native`

## Environment

| condition | GPU (cc) | kernel driver | libcuda (API) | CUDA toolkit | cuBLAS | CuPy | NumPy | emulation env | libcublas |
|---|---|---|---|---|---|---|---|---|---|
| cuda12 | NVIDIA GeForce RTX 5080 (12.0) | 590.48.01 | libcuda.so.590.48.01 (13.1) | 12.9.1 | 120901 | 14.2.0 | 2.2.6 | (unset) | /usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12.9.1.4<br>/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublasLt.so.12.9.1.4 |
| cuda13-native | NVIDIA GeForce RTX 5080 (12.0) | 590.48.01 | libcuda.so.590.48.01 (13.1) | 13.4.2 | 130800 | 14.2.0 | 2.2.6 | EMULATE_DOUBLE_PRECISION=0 | /usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublas.so.13.8.0.4<br>/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublasLt.so.13.8.0.4 |
| cuda13-emu | NVIDIA GeForce RTX 5080 (12.0) | 590.48.01 | libcuda.so.590.48.01 (13.1) | 13.4.2 | 130800 | 14.2.0 | 2.2.6 | EMULATE_DOUBLE_PRECISION=1, EMULATION_STRATEGY=performant | /usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublas.so.13.8.0.4<br>/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublasLt.so.13.8.0.4 |
| cuda13-emu-eager | NVIDIA GeForce RTX 5080 (12.0) | 590.48.01 | libcuda.so.590.48.01 (13.1) | 13.4.2 | 130800 | 14.2.0 | 2.2.6 | EMULATE_DOUBLE_PRECISION=1, EMULATION_STRATEGY=eager | /usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublas.so.13.8.0.4<br>/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcublasLt.so.13.8.0.4 |

Speedups are ratios of median times. Errors: *rel err vs NumPy* = max|C-C_np| / max|C_np|;
*scaled err* = max over sampled entries of |c_ij - exact_ij| / (|A||B|)_ij in units of u = 2^-53.

## DGEMM

![dgemm_tflops.png](figures/dgemm_tflops.png)
![dgemm_speedup.png](figures/dgemm_speedup.png)
![dgemm_error.png](figures/dgemm_error.png)
![dgemm_phi_tflops.png](figures/dgemm_phi_tflops.png)
![dgemm_phi_error.png](figures/dgemm_phi_error.png)

| phi | N | condition | median [s] | TFLOPS | speedup vs cuda12 | speedup vs cuda13-native | rel err vs NumPy | max scaled err [u] (GPU / NumPy) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2000 | cuda12 | 0.0203 | 0.790 | 1.00x | 1.00x | 3.76e-15 | 1.58 / 0.52 |
| 0 | 2000 | cuda13-native | 0.0203 | 0.787 | 1.00x | 1.00x | 3.76e-15 | 1.58 / 0.52 |
| 0 | 2000 | cuda13-emu | 0.0027 | 5.841 | 7.40x | 7.42x | 8.35e-16 | 0.13 / 0.52 |
| 0 | 2000 | cuda13-emu-eager | 0.0027 | 5.844 | 7.40x | 7.42x | 8.35e-16 | 0.13 / 0.52 |
| 0 | 4000 | cuda12 | 0.1581 | 0.809 | 1.00x | 1.02x | 5.24e-15 | 1.75 / 0.51 |
| 0 | 4000 | cuda13-native | 0.1619 | 0.791 | 0.98x | 1.00x | 5.24e-15 | 1.75 / 0.51 |
| 0 | 4000 | cuda13-emu | 0.0140 | 9.164 | 11.32x | 11.59x | 6.99e-16 | 0.13 / 0.51 |
| 0 | 4000 | cuda13-emu-eager | 0.0139 | 9.215 | 11.39x | 11.66x | 6.99e-16 | 0.13 / 0.51 |
| 0 | 8000 | cuda12 | 1.2519 | 0.818 | 1.00x | 1.01x | 7.80e-15 | 2.06 / 0.26 |
| 0 | 8000 | cuda13-native | 1.2608 | 0.812 | 0.99x | 1.00x | 7.80e-15 | 2.06 / 0.26 |
| 0 | 8000 | cuda13-emu | 0.0739 | 13.852 | 16.94x | 17.06x | 6.78e-16 | 0.06 / 0.26 |
| 0 | 8000 | cuda13-emu-eager | 0.0741 | 13.817 | 16.89x | 17.01x | 6.78e-16 | 0.06 / 0.26 |
| 0.5 | 2000 | cuda12 | 0.0203 | 0.790 | 1.00x | 1.00x | 4.08e-15 | 2.23 / 0.86 |
| 0.5 | 2000 | cuda13-native | 0.0203 | 0.787 | 1.00x | 1.00x | 4.08e-15 | 2.23 / 0.86 |
| 0.5 | 2000 | cuda13-emu | 0.0030 | 5.418 | 6.86x | 6.88x | 7.94e-16 | 0.10 / 0.86 |
| 0.5 | 2000 | cuda13-emu-eager | 0.0028 | 5.717 | 7.24x | 7.26x | 7.94e-16 | 0.10 / 0.86 |
| 0.5 | 4000 | cuda12 | 0.1582 | 0.809 | 1.00x | 1.02x | 4.95e-15 | 2.31 / 0.32 |
| 0.5 | 4000 | cuda13-native | 0.1619 | 0.791 | 0.98x | 1.00x | 4.95e-15 | 2.31 / 0.32 |
| 0.5 | 4000 | cuda13-emu | 0.0143 | 8.966 | 11.08x | 11.34x | 8.00e-16 | 0.10 / 0.32 |
| 0.5 | 4000 | cuda13-emu-eager | 0.0142 | 8.995 | 11.11x | 11.38x | 8.00e-16 | 0.10 / 0.32 |
| 0.5 | 8000 | cuda12 | 1.2519 | 0.818 | 1.00x | 1.01x | 8.19e-15 | 1.94 / 0.30 |
| 0.5 | 8000 | cuda13-native | 1.2609 | 0.812 | 0.99x | 1.00x | 8.19e-15 | 1.94 / 0.30 |
| 0.5 | 8000 | cuda13-emu | 0.0764 | 13.404 | 16.39x | 16.50x | 6.99e-16 | 0.05 / 0.30 |
| 0.5 | 8000 | cuda13-emu-eager | 0.0764 | 13.408 | 16.39x | 16.51x | 6.99e-16 | 0.05 / 0.30 |
| 1 | 2000 | cuda12 | 0.0203 | 0.790 | 1.00x | 1.00x | 2.22e-15 | 4.59 / 1.77 |
| 1 | 2000 | cuda13-native | 0.0203 | 0.787 | 1.00x | 1.00x | 2.22e-15 | 4.59 / 1.77 |
| 1 | 2000 | cuda13-emu | 0.0030 | 5.401 | 6.84x | 6.86x | 5.35e-16 | 0.32 / 1.77 |
| 1 | 2000 | cuda13-emu-eager | 0.0030 | 5.333 | 6.75x | 6.78x | 5.35e-16 | 0.32 / 1.77 |
| 1 | 4000 | cuda12 | 0.1582 | 0.809 | 1.00x | 1.02x | 6.71e-15 | 6.46 / 1.48 |
| 1 | 4000 | cuda13-native | 0.1619 | 0.791 | 0.98x | 1.00x | 6.71e-15 | 6.46 / 1.48 |
| 1 | 4000 | cuda13-emu | 0.0146 | 8.752 | 10.82x | 11.07x | 9.76e-16 | 0.17 / 1.48 |
| 1 | 4000 | cuda13-emu-eager | 0.0141 | 9.075 | 11.21x | 11.48x | 9.76e-16 | 0.17 / 1.48 |
| 1 | 8000 | cuda12 | 1.2522 | 0.818 | 1.00x | 1.01x | 6.52e-15 | 5.01 / 1.16 |
| 1 | 8000 | cuda13-native | 1.2608 | 0.812 | 0.99x | 1.00x | 6.52e-15 | 5.01 / 1.16 |
| 1 | 8000 | cuda13-emu | 0.0796 | 12.871 | 15.74x | 15.85x | 1.02e-15 | 0.18 / 1.16 |
| 1 | 8000 | cuda13-emu-eager | 0.0814 | 12.585 | 15.39x | 15.50x | 1.02e-15 | 0.18 / 1.16 |
| 2 | 2000 | cuda12 | 0.0203 | 0.790 | 1.00x | 1.00x | 8.11e-16 | 23.15 / 11.19 |
| 2 | 2000 | cuda13-native | 0.0203 | 0.787 | 1.00x | 1.00x | 8.11e-16 | 23.15 / 11.19 |
| 2 | 2000 | cuda13-emu | 0.0031 | 5.094 | 6.45x | 6.47x | 3.60e-16 | 1.20 / 11.19 |
| 2 | 2000 | cuda13-emu-eager | 0.0031 | 5.126 | 6.49x | 6.52x | 3.60e-16 | 1.20 / 11.19 |
| 2 | 4000 | cuda12 | 0.1586 | 0.807 | 1.00x | 1.02x | 3.99e-15 | 29.90 / 5.53 |
| 2 | 4000 | cuda13-native | 0.1619 | 0.790 | 0.98x | 1.00x | 3.99e-15 | 29.90 / 5.53 |
| 2 | 4000 | cuda13-emu | 0.0155 | 8.237 | 10.20x | 10.42x | 3.32e-16 | 1.31 / 5.53 |
| 2 | 4000 | cuda13-emu-eager | 0.0156 | 8.199 | 10.16x | 10.37x | 3.32e-16 | 1.31 / 5.53 |
| 2 | 8000 | cuda12 | 1.2556 | 0.816 | 1.00x | 1.00x | 7.50e-16 | 38.35 / 6.61 |
| 2 | 8000 | cuda13-native | 1.2607 | 0.812 | 1.00x | 1.00x | 7.50e-16 | 38.35 / 6.61 |
| 2 | 8000 | cuda13-emu | 0.0859 | 11.924 | 14.62x | 14.68x | 2.40e-16 | 1.24 / 6.61 |
| 2 | 8000 | cuda13-emu-eager | 0.0861 | 11.891 | 14.58x | 14.64x | 2.40e-16 | 1.24 / 6.61 |
| 4 | 2000 | cuda12 | 0.0203 | 0.789 | 1.00x | 1.00x | 7.13e-16 | 57.94 / 11.21 |
| 4 | 2000 | cuda13-native | 0.0203 | 0.787 | 1.00x | 1.00x | 7.13e-16 | 57.94 / 11.21 |
| 4 | 2000 | cuda13-emu | 0.0034 | 4.645 | 5.88x | 5.90x | 3.17e-16 | 1.93 / 11.21 |
| 4 | 2000 | cuda13-emu-eager | 0.0033 | 4.835 | 6.13x | 6.14x | 3.17e-16 | 1.93 / 11.21 |
| 4 | 4000 | cuda12 | 0.1586 | 0.807 | 1.00x | 1.02x | 1.04e-15 | 58.77 / 17.34 |
| 4 | 4000 | cuda13-native | 0.1619 | 0.791 | 0.98x | 1.00x | 1.04e-15 | 58.77 / 17.34 |
| 4 | 4000 | cuda13-emu | 0.0175 | 7.330 | 9.08x | 9.27x | 6.14e-16 | 1.82 / 17.34 |
| 4 | 4000 | cuda13-emu-eager | 0.0179 | 7.149 | 8.86x | 9.04x | 6.14e-16 | 1.82 / 17.34 |
| 4 | 8000 | cuda12 | 1.2556 | 0.816 | 1.00x | 1.00x | 2.12e-15 | 85.90 / 26.92 |
| 4 | 8000 | cuda13-native | 1.2572 | 0.815 | 1.00x | 1.00x | 2.12e-15 | 85.90 / 26.92 |
| 4 | 8000 | cuda13-emu | 0.1016 | 10.084 | 12.36x | 12.38x | 6.64e-17 | 1.80 / 26.92 |
| 4 | 8000 | cuda13-emu-eager | 0.1014 | 10.103 | 12.39x | 12.40x | 6.64e-17 | 1.80 / 26.92 |
