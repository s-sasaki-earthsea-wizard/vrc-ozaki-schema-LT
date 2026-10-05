#!/usr/bin/env bash
# Run every benchmark under every experimental condition, then build the report.
#
# Environment variables:
#   CONDITIONS   space-separated list (default: all four below)
#   QUICK=1      smoke test with small sizes (results go to results-quick/)
#   SKIP_BUILD=1 do not (re)build the images
#   SKIP_DGEMM=1 / SKIP_PDE=1
#   DGEMM_ARGS / PDE_ARGS  extra CLI arguments passed to the benchmark scripts
#
# Conditions:
#   cuda12            CUDA 12.x image, native FP64 DGEMM
#   cuda13-native     CUDA 13.4 image, emulation explicitly disabled
#   cuda13-emu        CUDA 13.4 image, emulation enabled, strategy "performant"
#                     (cuBLAS decides whether emulation pays off)
#   cuda13-emu-eager  CUDA 13.4 image, emulation enabled, strategy "eager"
#                     (emulate whenever possible)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONDITIONS="${CONDITIONS:-cuda12 cuda13-native cuda13-emu cuda13-emu-eager}"
DGEMM_ARGS="${DGEMM_ARGS:-}"
PDE_ARGS="${PDE_ARGS:-}"
RESULTS_DIR="results"

if [[ "${QUICK:-0}" == "1" ]]; then
    RESULTS_DIR="results-quick"
    DGEMM_ARGS="--sizes 512 1024 2048 --repeats 3 --samples 64 ${DGEMM_ARGS}"
    PDE_ARGS="--sizes 256 512 1024 --adi-steps 5 --ftcs-steps 100 --repeats 2 ${PDE_ARGS}"
fi

export HOST_UID HOST_GID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# condition -> "service ENV=VALUE ..."
condition_spec() {
    case "$1" in
        cuda12)           echo "cuda12" ;;
        cuda13-native)    echo "cuda13 CUBLAS_EMULATE_DOUBLE_PRECISION=0" ;;
        cuda13-emu)       echo "cuda13 CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=performant" ;;
        cuda13-emu-eager) echo "cuda13 CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager" ;;
        *) echo "unknown condition: $1" >&2; return 1 ;;
    esac
}

run_in() {
    # run_in <condition> <command...>
    local cond="$1"; shift
    local spec service
    spec="$(condition_spec "${cond}")"
    read -r service envs <<<"${spec} "
    local env_flags=(-e "BENCH_CONDITION=${cond}")
    for kv in ${envs}; do env_flags+=(-e "${kv}"); done
    docker compose run --rm "${env_flags[@]}" "${service}" "$@"
}

mkdir -p "${RESULTS_DIR}" cache

services=()
for cond in ${CONDITIONS}; do
    read -r svc _ <<<"$(condition_spec "${cond}")"
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
    run_in "${cond}" python src/common.py | tee "${out}/env.json"

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
docker compose run --rm -e BENCH_CONDITION=analysis cuda13 \
    python src/analyze_results.py --results-dir "${RESULTS_DIR}" --conditions ${CONDITIONS}

log "done: ${RESULTS_DIR}/report.md"
