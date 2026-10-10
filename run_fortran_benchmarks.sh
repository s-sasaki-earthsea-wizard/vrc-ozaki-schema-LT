#!/usr/bin/env bash
# Fortran drop-in experiment: one unmodified Fortran DGEMM binary (fortran/dgemm_bench.f90,
# linked against OpenBLAS) run under three conditions without recompiling:
#
#   fortran-openblas       OpenBLAS on the CPU
#   fortran-nvblas-native  LD_PRELOAD=libnvblas.so -> cuBLAS, FP64 emulation off
#   fortran-nvblas-emu     LD_PRELOAD=libnvblas.so -> cuBLAS, FP64 emulation on (Ozaki)
#
# NVBLAS copies the operands host -> device and the result back on every call, so the
# GPU timings are end to end and include the PCIe transfers.
#
# NVBLAS fails silently: if cublasXtDgemm fails (e.g. CUBLAS_STATUS_ALLOC_FAILED with
# NVBLAS_TILE_DIM 16384 on a 16 GB GPU), dgemm_ returns without touching C and only
# NVBLAS_LOGFILE says so. Each run is therefore checked against that log (and the
# sampled accuracy check would show errors of ~1e14 u).
#
# Environment variables:
#   RESULTS_DIR   output directory (default: results-fortran, or results-quick/fortran with QUICK=1)
#   SIZES         default: "4000 8000 16000"
#   PHI           default: 0.5
#   REPEATS       timed DGEMM calls after one warm-up call (default: 3)
#   SAMPLES       entries checked against a real128 reference (default: 64)
#   THREADS       OpenBLAS threads for the CPU condition (default: 14 = P+E, fastest in the sweep)
#   TILE_DIMS     NVBLAS_TILE_DIM values to try (default: "2048 8192")
#   AUTOPIN=1     enable NVBLAS_AUTOPIN_MEM_ENABLED (slower here: pinning costs more than it saves)
#   PROFILE_N     size for the nsys kernel check of the emulated run (default: 8000, empty = skip)
#   QUICK=1       smoke test with small sizes
#   SKIP_BUILD=1  do not (re)build the docker image
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

SIZES="${SIZES:-4000 8000 16000}"
PHI="${PHI:-0.5}"
REPEATS="${REPEATS:-3}"
SAMPLES="${SAMPLES:-64}"
THREADS="${THREADS:-14}"
TILE_DIMS="${TILE_DIMS:-2048 8192}"
PROFILE_N="${PROFILE_N-8000}"
if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-results-quick/fortran}"
    SIZES="1000 2000"
    REPEATS=2
    PROFILE_N=2000
fi
RESULTS_DIR="${RESULTS_DIR:-results-fortran}"

BUILD_DIR=build/fortran
BIN="${BUILD_DIR}/dgemm_bench"
NVBLAS_LIB=/usr/local/cuda/lib64/libnvblas.so.13
OUT="${RESULTS_DIR}/dgemm.txt"
NVBLAS_LOG="${BUILD_DIR}/nvblas.log"

fortran_run() { docker compose run --rm -T "$@"; }

# write_nvblas_conf <tile> -> path of the config file (inside the container)
write_nvblas_conf() {
    local conf="${BUILD_DIR}/nvblas_tile$1.conf"
    {
        echo "NVBLAS_LOGFILE /workspace/${NVBLAS_LOG}"
        echo "NVBLAS_CPU_BLAS_LIB /usr/lib/x86_64-linux-gnu/libopenblas.so.0"
        echo "NVBLAS_GPU_LIST ALL"
        echo "NVBLAS_TILE_DIM $1"
        if [[ "${AUTOPIN:-0}" == "1" ]]; then echo "NVBLAS_AUTOPIN_MEM_ENABLED"; fi
    } > "${conf}"
    echo "/workspace/${conf}"
}

# record <label> <n> <docker compose run args...>: run the binary and append its RESULT
# line, plus a FAILED line if NVBLAS logged a failed call during the run
record() {
    local label="$1" n="$2"; shift 2
    log "${label} n=${n}"
    rm -f "${NVBLAS_LOG}"
    fortran_run "$@" "${BIN}" "${n}" "${PHI}" "${REPEATS}" "${SAMPLES}" 2>&1 \
        | grep '^RESULT' | sed "s/^RESULT /RESULT ${label} /" | tee -a "${OUT}"
    if [[ -f "${NVBLAS_LOG}" ]] && grep -q 'failed' "${NVBLAS_LOG}"; then
        echo "FAILED ${label} n=${n}: $(grep 'failed' "${NVBLAS_LOG}" | sort | uniq -c | xargs)" | tee -a "${OUT}"
    fi
}

mkdir -p "${RESULTS_DIR}" "${BUILD_DIR}"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "building the fortran image (on top of vrc-ozaki/cuda13)"
    docker compose build fortran
fi

log "compiling fortran/"
fortran_run fortran gfortran -O2 -march=native -Wall -o "${BIN}" fortran/dgemm_bench.f90 -lopenblas

{
    echo "# Fortran DGEMM drop-in: $(date -Iseconds)"
    echo "# phi=${PHI} repeats=${REPEATS} samples=${SAMPLES} openblas_threads=${THREADS} autopin=${AUTOPIN:-0}"
    # Prefix the lines we want: the CUDA image entrypoint prints a banner to stdout.
    fortran_run fortran bash -c '
        echo "ENV $(gfortran --version | head -1)"
        echo "ENV gpu, driver, pcie gen max, pcie width: $(nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.current --format=csv,noheader)"
    ' 2>/dev/null | grep '^ENV ' | sed 's/^ENV /# /'
} > "${OUT}"

for n in ${SIZES}; do
    record "condition=fortran-openblas threads=${THREADS}" "${n}" -e "OPENBLAS_NUM_THREADS=${THREADS}" fortran
    for tile in ${TILE_DIMS}; do
        conf="$(write_nvblas_conf "${tile}")"
        for emu in 0 1; do
            if [[ "${emu}" == "1" ]]; then cond=fortran-nvblas-emu; else cond=fortran-nvblas-native; fi
            record "condition=${cond} tile=${tile}" "${n}" \
                -e "LD_PRELOAD=${NVBLAS_LIB}" -e "NVBLAS_CONFIG_FILE=${conf}" \
                -e "CUBLAS_EMULATE_DOUBLE_PRECISION=${emu}" -e CUBLAS_EMULATION_STRATEGY=performant \
                fortran
        done
    done
done

if [[ -n "${PROFILE_N}" ]]; then
    # Which kernels did cuBLAS run behind NVBLAS? No NVTX range here, so the summary covers
    # all calls (warm-up included); the env goes to the profiled process only (nsys -e),
    # not to nsys itself.
    tile="${TILE_DIMS##* }"
    conf="$(write_nvblas_conf "${tile}")"
    rep="${RESULTS_DIR}/nsys/fortran-nvblas-emu/n${PROFILE_N}_phi${PHI}"
    mkdir -p "$(dirname "${rep}")"
    log "nsys: fortran-nvblas-emu n=${PROFILE_N} tile=${tile}"
    fortran_run fortran bash -c "
        nsys profile -t cuda,cublas --force-overwrite true -o '${rep}' \
            -e LD_PRELOAD=${NVBLAS_LIB},NVBLAS_CONFIG_FILE=${conf},CUBLAS_EMULATE_DOUBLE_PRECISION=1,CUBLAS_EMULATION_STRATEGY=performant \
            ${BIN} ${PROFILE_N} ${PHI} ${REPEATS} 0 > /dev/null &&
        for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
            nsys stats -q --report \$r --format csv --force-export true --force-overwrite true -o '${rep}' '${rep}.nsys-rep' > /dev/null
        done &&
        rm -f '${rep}.sqlite'
    "
    calls=$((REPEATS + 1))
    fortran_run fortran python - <<EOF | grep '^NSYS ' | tee -a "${OUT}"
import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
from summarize_nsys import load_case

rep = "${rep}"
case = load_case(Path(f"{rep}_cuda_gpu_kern_sum.csv"), ${calls})
with open(f"{rep}_cuda_gpu_mem_time_sum.csv", newline="", encoding="utf-8") as f:
    copy_ms = sum(float(r["Total Time (ns)"]) for r in csv.DictReader(f) if "memcpy" in r["Operation"]) / ${calls} / 1e6
print(
    f"NSYS condition=fortran-nvblas-emu n=${PROFILE_N} tile=${tile} path={case['path']} "
    f"kernel_ms_per_dgemm={case['gpu_time_per_dgemm_ms']:.1f} memcpy_ms_per_dgemm={copy_ms:.1f}"
)
EOF
fi

log "done: ${OUT}"
