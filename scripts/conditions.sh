# Shared definition of the experimental conditions. Source this file.
#
#   cuda12            CUDA 12.x image, native FP64 DGEMM
#   cuda13-native     CUDA 13.4 image, emulation explicitly disabled
#   cuda13-emu        CUDA 13.4 image, emulation enabled, strategy "performant"
#                     (cuBLAS decides whether emulation pays off)
#   cuda13-emu-eager  CUDA 13.4 image, emulation enabled, strategy "eager"
#                     (emulate whenever possible)

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

condition_service() {
    local spec
    spec="$(condition_spec "$1")"
    echo "${spec%% *}"
}

# run_in <condition> <command...>
run_in() {
    local cond="$1"; shift
    local spec service envs
    spec="$(condition_spec "${cond}")"
    read -r service envs <<<"${spec} "
    local env_flags=(-e "BENCH_CONDITION=${cond}")
    for kv in ${envs}; do env_flags+=(-e "${kv}"); done
    docker compose run --rm -T "${env_flags[@]}" "${service}" "$@"
}
