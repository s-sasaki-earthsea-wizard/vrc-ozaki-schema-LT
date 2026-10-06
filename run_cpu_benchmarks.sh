#!/usr/bin/env bash
# CPU baselines: build cpp/ (Eigen + OpenMP) inside the cpu image, then run the DGEMM and
# PDE benchmarks with a thread sweep and rebuild the report including the GPU results.
#
# Environment variables:
#   RESULTS_DIR   output directory (default: results, or results-quick with QUICK=1)
#   THREADS       thread counts to sweep (default: "6 14 16" = P / P+E / all cores)
#   QUICK=1       smoke test with small sizes
#   SKIP_BUILD=1  do not (re)build the docker image
#   SKIP_DGEMM=1 / SKIP_PDE=1
#   DGEMM_ARGS / PDE_ARGS  extra CLI arguments for src/benchmark_cpu.py
#
# Do not run this concurrently with the GPU benchmarks: both use the host CPU.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

THREADS="${THREADS:-6 14 16}"
DGEMM_ARGS="${DGEMM_ARGS:-}"
PDE_ARGS="${PDE_ARGS:-}"
if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-results-quick}"
    DGEMM_ARGS="--sizes 512 1024 2048 --repeats 2 --samples 64 ${DGEMM_ARGS}"
    PDE_ARGS="--sizes 256 512 1024 --adi-steps 5 --ftcs-steps 100 ${PDE_ARGS}"
fi
RESULTS_DIR="${RESULTS_DIR:-results}"

cpu_run() { docker compose run --rm -T -e BENCH_CONDITION=cpu cpu "$@"; }

mkdir -p "${RESULTS_DIR}" cache
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "building the cpu image"
    docker compose build cpu
fi

log "compiling cpp/"
cpu_run bash -c "cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build/cpp -j"
cpu_run build/cpp/cpu_bench version

for cond in cpu-eigen cpu-numpy; do mkdir -p "${RESULTS_DIR}/${cond}"; done

if [[ "${SKIP_DGEMM:-0}" != "1" ]]; then
    log "=== CPU DGEMM (threads: ${THREADS}) ==="
    # shellcheck disable=SC2086
    cpu_run python src/benchmark_cpu.py dgemm --results-dir "${RESULTS_DIR}" --threads ${THREADS} ${DGEMM_ARGS} \
        2>&1 | tee "${RESULTS_DIR}/cpu-eigen/dgemm.log"
fi

if [[ "${SKIP_PDE:-0}" != "1" ]]; then
    log "=== CPU PDE (threads: ${THREADS}) ==="
    # shellcheck disable=SC2086
    cpu_run python src/benchmark_cpu.py pde --results-dir "${RESULTS_DIR}" --threads ${THREADS} ${PDE_ARGS} \
        2>&1 | tee "${RESULTS_DIR}/cpu-eigen/pde.log"
fi

rm -rf cache/cpu_work

log "=== analysis ==="
cpu_run python src/analyze_results.py --results-dir "${RESULTS_DIR}"
log "done: ${RESULTS_DIR}/report.md"
