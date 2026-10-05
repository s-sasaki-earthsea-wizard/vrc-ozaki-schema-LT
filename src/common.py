"""Shared helpers for the benchmarks: environment capture, GPU timing, JSON I/O.

Running this module directly prints the environment report, which is used as a
pre-flight check (is the GPU visible, which cuBLAS is actually loaded, is FP64
emulation requested).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

EMULATION_ENV_VARS = (
    "CUBLAS_EMULATE_DOUBLE_PRECISION",
    "CUBLAS_EMULATION_STRATEGY",
    "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
    "CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK",
)

UNIT_ROUNDOFF = 2.0**-53


def condition_name() -> str:
    """Return the experimental condition label (set by run_benchmarks.sh)."""
    return os.environ.get("BENCH_CONDITION", "unlabeled")


def _cuda_version_str(v: int) -> str:
    """Convert an integer CUDA version (e.g. 13040) to 'major.minor'."""
    return f"{v // 1000}.{(v % 1000) // 10}"


def _loaded_libraries(pattern: str) -> list[str]:
    """List shared libraries mapped into this process whose path contains pattern."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            paths = {line.split()[-1] for line in f if pattern in line and "/" in line}
        return sorted(paths)
    except OSError:
        return []


def _nvidia_smi_snapshot() -> dict[str, str]:
    """Query a few GPU state fields via nvidia-smi (best effort)."""
    fields = [
        "name",
        "driver_version",
        "clocks.sm",
        "clocks.max.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "persistence_mode",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip().splitlines()[0]
        return dict(zip(fields, (v.strip() for v in out.split(","))))
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}


def warm_up_cublas() -> None:
    """Run a tiny DGEMM so that the cuBLAS handle and library are loaded."""
    import cupy as cp

    a = cp.ones((64, 64), dtype=cp.float64)
    (a @ a).sum().item()


def collect_env_info() -> dict[str, Any]:
    """Collect software/hardware information needed to interpret the results.

    Returns:
        A JSON-serialisable dictionary.
    """
    import cupy as cp

    warm_up_cublas()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)

    cublas_version = None
    try:
        from cupy_backends.cuda.libs import cublas

        cublas_version = cublas.getVersion(dev.cublas_handle)
    except Exception as exc:  # noqa: BLE001 - informational only
        cublas_version = f"unavailable ({exc!r})"

    try:
        blas_info = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except Exception:  # noqa: BLE001 - informational only
        blas_info = {}

    return {
        "condition": condition_name(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numpy_blas": blas_info,
        "cupy": cp.__version__,
        # CuPy wheels link cudart statically: this is CuPy's build-time version,
        # not the toolkit in the image (see cublas_version / libcublas_loaded).
        "cuda_runtime": _cuda_version_str(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_toolkit": os.environ.get("CUDA_VERSION"),
        "cuda_driver_api": _cuda_version_str(cp.cuda.runtime.driverGetVersion()),
        "cublas_version": cublas_version,
        "libcublas_loaded": _loaded_libraries("libcublas"),
        "libcuda_loaded": _loaded_libraries("libcuda.so"),
        "gpu_name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
        "compute_capability": f"{props['major']}.{props['minor']}",
        "gpu_memory_gib": round(props["totalGlobalMem"] / 2**30, 2),
        "nvidia_smi": _nvidia_smi_snapshot(),
        "emulation_env": {k: os.environ.get(k) for k in EMULATION_ENV_VARS},
        "cpu_count": os.cpu_count(),
    }


def time_gpu(fn: Callable[[], None], warmup: int, repeats: int) -> list[float]:
    """Time a GPU workload with CUDA events.

    Args:
        fn: Callable that enqueues the workload on the current stream.
        warmup: Number of untimed calls.
        repeats: Number of timed calls.

    Returns:
        Elapsed seconds of each timed call.
    """
    import cupy as cp

    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop) / 1e3)
    return times


def summarize_times(times: list[float]) -> dict[str, float]:
    """Return median/min/max/std of a list of timings."""
    arr = np.asarray(times)
    return {
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def free_gpu_memory() -> None:
    """Release cached blocks of the CuPy memory pools."""
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write payload as pretty-printed JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log(msg: str) -> None:
    """Print a timestamped progress message."""
    now = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{condition_name()}] {msg}", flush=True)


if __name__ == "__main__":
    print(json.dumps(collect_env_info(), indent=2, ensure_ascii=False))
