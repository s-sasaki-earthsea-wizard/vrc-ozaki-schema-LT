# DGEMM kernels observed with Nsight Systems

Kernels inside the NVTX range `dgemm` (steady state, excluding warm-up and transfers).
*path* is derived from kernel names: `oz2_*` = Ozaki-II, `d884gemm` = native FP64 (DMMA).
*INT8 MMA ms* = time in INT8 tensor-core GEMMs per DGEMM; these are batched over the moduli,
so at fixed N it grows with the number of moduli that cuBLAS selects.

| condition | N | phi | path | GPU ms / DGEMM | INT8 MMA ms | INT8 MMA share | top kernels (time %) |
|---|---:|---:|---|---:|---:|---:|---|
| cuda13-native | 512 | 0.5 | native FP64 | 0.436 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 512 | 0.5 | native FP64 | 0.434 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu-eager | 512 | 0.5 | Ozaki-II | 0.178 | 0.032 | 18% | `prolog_mod_dfma<oz2_int8_dgemm,...>` (34.5%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (28.7%)<br>`nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (17.9%)<br>`get_minmax_block_ker<,...>` (4.1%) |
| cuda13-native | 2048 | 0.5 | native FP64 | 22.434 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 2048 | 0.5 | Ozaki-II | 2.668 | 0.809 | 30% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (30.3%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (29.4%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (24.7%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (3.7%) |
| cuda13-emu-eager | 2048 | 0.5 | Ozaki-II | 2.667 | 0.808 | 30% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (30.3%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (29.4%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (24.7%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (3.7%) |
| cuda13-native | 8000 | 0.5 | native FP64 | 1260.799 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 8000 | 0.5 | Ozaki-II | 71.740 | 42.080 | 59% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (58.7%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (16.0%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (13.6%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (2.7%) |
| cuda13-emu-eager | 8000 | 0.5 | Ozaki-II | 71.747 | 42.106 | 59% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (58.7%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (16.0%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (13.6%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (2.7%) |
| cuda13-native | 512 | 4 | native FP64 | 0.436 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 512 | 4 | native FP64 | 0.434 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu-eager | 512 | 4 | Ozaki-II | 0.193 | 0.042 | 22% | `prolog_mod_dfma<oz2_int8_dgemm,...>` (31.9%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (27.8%)<br>`nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (21.6%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (4.1%) |
| cuda13-native | 2048 | 4 | native FP64 | 22.432 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 2048 | 4 | Ozaki-II | 3.180 | 1.259 | 40% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (39.6%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (24.7%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (21.3%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (4.5%) |
| cuda13-emu-eager | 2048 | 4 | Ozaki-II | 3.179 | 1.259 | 40% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (39.6%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (24.6%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (21.3%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (4.5%) |
| cuda13-native | 8000 | 4 | native FP64 | 1260.565 | 0.000 | 0% | `cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x64_16x4_nn_align1,...` (100.0%) |
| cuda13-emu | 8000 | 4 | Ozaki-II | 98.651 | 67.934 | 69% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (68.9%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (11.6%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (10.1%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (2.8%) |
| cuda13-emu-eager | 8000 | 4 | Ozaki-II | 98.461 | 67.741 | 69% | `nvjet_sm120_biu_mma_256x128x128_2_64x64x128_tmaAB_alignCD4_bz_dynba...` (68.8%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (11.7%)<br>`prolog_mod_dfma<oz2_int8_dgemm,...>` (10.1%)<br>`inverse_scaling_mrc_kernel<oz2_int8_dgemm,...>` (2.8%) |
