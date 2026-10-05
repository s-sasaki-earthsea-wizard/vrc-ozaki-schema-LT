#!/usr/bin/env bash
# Run every benchmark under every experimental condition, then build the report.
#
# Environment variables:
#   CONDITIONS   space-separated list (default: all four, see scripts/conditions.sh)
#   RESULTS_DIR  output directory (default: results, or results-quick with QUICK=1)
#   QUICK=1      smoke test with small sizes
#   SKIP_BUILD=1 do not (re)build the images
#   SKIP_DGEMM=1 / SKIP_PDE=1
#   DGEMM_ARGS / PDE_ARGS  extra CLI arguments passed to the benchmark scripts
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/conditions.sh
source scripts/conditions.sh

CONDITIONS="${CONDITIONS:-cuda12 cuda13-native cuda13-emu cuda13-emu-eager}"
DGEMM_ARGS="${DGEMM_ARGS:-}"
PDE_ARGS="${PDE_ARGS:-}"

if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-results-quick}"
    DGEMM_ARGS="--sizes 512 1024 2048 --repeats 3 --samples 64 ${DGEMM_ARGS}"
    PDE_ARGS="--sizes 256 512 1024 --adi-steps 5 --ftcs-steps 100 --repeats 2 ${PDE_ARGS}"
fi
RESULTS_DIR="${RESULTS_DIR:-results}"

mkdir -p "${RESULTS_DIR}" cache

services=()
for cond in ${CONDITIONS}; do
    svc="$(condition_service "${cond}")"
    [[ " ${services[*]} " == *" ${svc} "* ]] || services+=("${svc}")
done

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "building images: ${services[*]}"
    docker compose build "${services[@]}"
fi

log "host GPU state"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv

for cond in ${CONDITIONS}; do
    out="${RESULTS_DIR}/${cond}"
    mkdir -p "${out}"
    log "=== ${cond}: environment check ==="
    run_in "${cond}" python src/common.py "${out}/env.json"

    if [[ "${SKIP_DGEMM:-0}" != "1" ]]; then
        log "=== ${cond}: DGEMM ==="
        # shellcheck disable=SC2086
        run_in "${cond}" python src/benchmark_dgemm.py --output "${out}/dgemm.json" ${DGEMM_ARGS} \
            2>&1 | tee "${out}/dgemm.log"
    fi

    if [[ "${SKIP_PDE:-0}" != "1" ]]; then
        log "=== ${cond}: PDE ==="
        # shellcheck disable=SC2086
        run_in "${cond}" python src/benchmark_pde.py --output "${out}/pde.json" ${PDE_ARGS} \
            2>&1 | tee "${out}/pde.log"
    fi
done

log "=== analysis ==="
# shellcheck disable=SC2086
run_in cuda13-native python src/analyze_results.py --results-dir "${RESULTS_DIR}" --conditions ${CONDITIONS}

log "done: ${RESULTS_DIR}/report.md"
