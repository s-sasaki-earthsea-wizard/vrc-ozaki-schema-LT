# CUDA 12 vs CUDA 13.4 (Ozaki scheme) FP64 benchmark report

- conditions: cuda12, cuda13-native, cuda13-emu, cuda13-emu-eager, nsys
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

| phi | N | condition | median [s] | TFLOPS | speedup vs cuda12 | speedup vs cuda13-native | rel err vs NumPy | max scaled err [u] (GPU / NumPy) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 2000 | cuda12 | 0.0203 | 0.790 | 1.00x | 1.00x | 4.08e-15 | 2.23 / 0.86 |
| 0.5 | 2000 | cuda13-native | 0.0202 | 0.792 | 1.00x | 1.00x | 4.08e-15 | 2.23 / 0.86 |
| 0.5 | 2000 | cuda13-emu | 0.0029 | 5.458 | 6.91x | 6.89x | 7.94e-16 | 0.10 / 0.86 |
| 0.5 | 2000 | cuda13-emu-eager | 0.0028 | 5.716 | 7.24x | 7.22x | 7.94e-16 | 0.10 / 0.86 |
| 0.5 | 4000 | cuda12 | 0.1581 | 0.810 | 1.00x | 1.02x | 4.95e-15 | 2.31 / 0.32 |
| 0.5 | 4000 | cuda13-native | 0.1613 | 0.793 | 0.98x | 1.00x | 4.95e-15 | 2.31 / 0.32 |
| 0.5 | 4000 | cuda13-emu | 0.0142 | 9.015 | 11.13x | 11.36x | 8.00e-16 | 0.10 / 0.32 |
| 0.5 | 4000 | cuda13-emu-eager | 0.0143 | 8.964 | 11.07x | 11.30x | 8.00e-16 | 0.10 / 0.32 |
| 0.5 | 8000 | cuda12 | 1.2553 | 0.816 | 1.00x | 1.00x | 8.19e-15 | 1.94 / 0.30 |
| 0.5 | 8000 | cuda13-native | 1.2565 | 0.815 | 1.00x | 1.00x | 8.19e-15 | 1.94 / 0.30 |
| 0.5 | 8000 | cuda13-emu | 0.0766 | 13.372 | 16.39x | 16.41x | 6.99e-16 | 0.05 / 0.30 |
| 0.5 | 8000 | cuda13-emu-eager | 0.0763 | 13.415 | 16.45x | 16.46x | 6.99e-16 | 0.05 / 0.30 |
| 0.5 | 16000 | cuda12 | 10.0336 | 0.816 | 1.00x | 1.00x | 1.34e-14 | 1.90 / 0.20 |
| 0.5 | 16000 | cuda13-native | 10.0203 | 0.818 | 1.00x | 1.00x | 1.34e-14 | 1.90 / 0.20 |
| 0.5 | 16000 | cuda13-emu | 0.5239 | 15.637 | 19.15x | 19.13x | 8.94e-16 | 0.05 / 0.20 |
| 0.5 | 16000 | cuda13-emu-eager | 0.5234 | 15.650 | 19.17x | 19.14x | 8.94e-16 | 0.05 / 0.20 |

## 2D heat equation

![pde_time_per_step.png](figures/pde_time_per_step.png)
![pde_speedup.png](figures/pde_speedup.png)

| scheme | N | steps | condition | ms/step | TFLOPS | speedup vs cuda12 | speedup vs cuda13-native | rel err vs discrete exact | rel err vs NumPy | discretization err |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| adi-gemm | 1024 | 20 | cuda12 | 6.021 | 0.7133 | 1.00x | 1.15x | 2.47e-14 | 3.02e-14 | 5.40e-06 |
| adi-gemm | 1024 | 20 | cuda13-native | 6.894 | 0.6230 | 0.87x | 1.00x | 2.47e-14 | 3.02e-14 | 5.40e-06 |
| adi-gemm | 1024 | 20 | cuda13-emu | 1.429 | 3.0059 | 4.21x | 4.83x | 7.12e-15 | 1.11e-14 | 5.40e-06 |
| adi-gemm | 1024 | 20 | cuda13-emu-eager | 1.429 | 3.0059 | 4.21x | 4.83x | 7.12e-15 | 1.11e-14 | 5.40e-06 |
| adi-gemm | 2048 | 20 | cuda12 | 43.008 | 0.7989 | 1.00x | 1.04x | 6.04e-14 | 5.72e-14 | 7.02e-06 |
| adi-gemm | 2048 | 20 | cuda13-native | 44.858 | 0.7660 | 0.96x | 1.00x | 6.04e-14 | 5.72e-14 | 7.02e-06 |
| adi-gemm | 2048 | 20 | cuda13-emu | 5.708 | 6.0198 | 7.53x | 7.86x | 3.77e-15 | 1.53e-14 | 7.02e-06 |
| adi-gemm | 2048 | 20 | cuda13-emu-eager | 5.709 | 6.0182 | 7.53x | 7.86x | 3.77e-15 | 1.53e-14 | 7.02e-06 |
| adi-gemm | 4096 | 20 | cuda12 | 337.141 | 0.8153 | 1.00x | 1.00x | 1.14e-13 | - | 7.42e-06 |
| adi-gemm | 4096 | 20 | cuda13-native | 338.664 | 0.8117 | 1.00x | 1.00x | 1.14e-13 | - | 7.42e-06 |
| adi-gemm | 4096 | 20 | cuda13-emu | 29.238 | 9.4015 | 11.53x | 11.58x | 6.42e-15 | - | 7.42e-06 |
| adi-gemm | 4096 | 20 | cuda13-emu-eager | 29.258 | 9.3951 | 11.52x | 11.58x | 6.42e-15 | - | 7.42e-06 |
| adi-gemm | 8192 | 20 | cuda12 | 2692.224 | 0.8168 | 1.00x | 1.00x | 1.98e-13 | - | 7.52e-06 |
| adi-gemm | 8192 | 20 | cuda13-native | 2697.351 | 0.8153 | 1.00x | 1.00x | 1.98e-13 | - | 7.52e-06 |
| adi-gemm | 8192 | 20 | cuda13-emu | 169.782 | 12.9520 | 15.86x | 15.89x | 7.26e-15 | - | 7.52e-06 |
| adi-gemm | 8192 | 20 | cuda13-emu-eager | 169.880 | 12.9446 | 15.85x | 15.88x | 7.26e-15 | - | 7.52e-06 |
| ftcs-stencil | 1024 | 1000 | cuda12 | 0.016 | 0.4475 | 1.00x | 1.00x | 3.32e-14 | 1.66e-16 | 8.94e-07 |
| ftcs-stencil | 1024 | 1000 | cuda13-native | 0.016 | 0.4473 | 1.00x | 1.00x | 3.32e-14 | 1.66e-16 | 8.94e-07 |
| ftcs-stencil | 1024 | 1000 | cuda13-emu | 0.016 | 0.4475 | 1.00x | 1.00x | 3.32e-14 | 1.66e-16 | 8.94e-07 |
| ftcs-stencil | 1024 | 1000 | cuda13-emu-eager | 0.016 | 0.4475 | 1.00x | 1.00x | 3.32e-14 | 1.66e-16 | 8.94e-07 |
| ftcs-stencil | 2048 | 1000 | cuda12 | 0.057 | 0.5112 | 1.00x | 1.00x | 1.39e-14 | 1.64e-16 | 6.19e-08 |
| ftcs-stencil | 2048 | 1000 | cuda13-native | 0.057 | 0.5114 | 1.00x | 1.00x | 1.39e-14 | 1.64e-16 | 6.19e-08 |
| ftcs-stencil | 2048 | 1000 | cuda13-emu | 0.058 | 0.5065 | 0.99x | 0.99x | 1.39e-14 | 1.64e-16 | 6.19e-08 |
| ftcs-stencil | 2048 | 1000 | cuda13-emu-eager | 0.057 | 0.5114 | 1.00x | 1.00x | 1.39e-14 | 1.64e-16 | 6.19e-08 |
| ftcs-stencil | 4096 | 1000 | cuda12 | 0.331 | 0.3551 | 1.00x | 1.00x | 3.88e-14 | - | 3.97e-09 |
| ftcs-stencil | 4096 | 1000 | cuda13-native | 0.331 | 0.3550 | 1.00x | 1.00x | 3.88e-14 | - | 3.97e-09 |
| ftcs-stencil | 4096 | 1000 | cuda13-emu | 0.331 | 0.3551 | 1.00x | 1.00x | 3.88e-14 | - | 3.97e-09 |
| ftcs-stencil | 4096 | 1000 | cuda13-emu-eager | 0.331 | 0.3551 | 1.00x | 1.00x | 3.88e-14 | - | 3.97e-09 |
| ftcs-stencil | 8192 | 1000 | cuda12 | 1.324 | 0.3547 | 1.00x | 1.00x | 1.71e-14 | - | 2.50e-10 |
| ftcs-stencil | 8192 | 1000 | cuda13-native | 1.324 | 0.3548 | 1.00x | 1.00x | 1.71e-14 | - | 2.50e-10 |
| ftcs-stencil | 8192 | 1000 | cuda13-emu | 1.324 | 0.3547 | 1.00x | 1.00x | 1.71e-14 | - | 2.50e-10 |
| ftcs-stencil | 8192 | 1000 | cuda13-emu-eager | 1.324 | 0.3547 | 1.00x | 1.00x | 1.71e-14 | - | 2.50e-10 |
