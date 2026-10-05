#!/usr/bin/env bash
# Profile DGEMM with Nsight Systems to see which kernels cuBLAS actually runs
# (Ozaki-II / Ozaki-I emulation vs native FP64), then summarize.
#
# Environment variables:
#   CONDITIONS  default: the CUDA 13.4 conditions (nsys ships only in that image)
#   SIZES       default: "512 2048 8000"
#   PHIS        default: "0.5 4"
#   ITERS       DGEMM calls inside the profiled NVTX range (default: 3)
#   OUT_DIR     default: results/nsys
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

CONDITIONS="${CONDITIONS:-cuda13-native cuda13-emu cuda13-emu-eager}"
SIZES="${SIZES:-512 2048 8000}"
PHIS="${PHIS:-0.5 4}"
ITERS="${ITERS:-3}"
OUT_DIR="${OUT_DIR:-results/nsys}"

for cond in ${CONDITIONS}; do
    if [[ "$(condition_service "${cond}")" != "cuda13" ]]; then
        log "skip ${cond}: nsys is only available in the cuda13 image"
        continue
    fi
    mkdir -p "${OUT_DIR}/${cond}"
    for phi in ${PHIS}; do
        for n in ${SIZES}; do
            rep="${OUT_DIR}/${cond}/n${n}_phi${phi}"
            log "=== ${cond}: n=${n} phi=${phi} ==="
            run_in "${cond}" bash -c "
                nsys profile -t cuda,nvtx,cublas --force-overwrite true -o '${rep}' \
                    python src/profile_dgemm.py --n ${n} --phi ${phi} --iters ${ITERS} > /dev/null &&
                nsys stats -q --report cuda_gpu_kern_sum --format csv --filter-nvtx dgemm \
                    --force-export true --force-overwrite true -o '${rep}' '${rep}.nsys-rep' > /dev/null &&
                rm -f '${rep}.sqlite'
            "
        done
    done
done

run_in cuda13-native python src/summarize_nsys.py --dir "${OUT_DIR}" --iters "${ITERS}"
log "done: ${OUT_DIR}/kernels.md"
