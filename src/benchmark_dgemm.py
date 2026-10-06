"""FP64 dense matrix multiplication (DGEMM) benchmark: C = A @ B.

Measures time / TFLOPS on the GPU (CuPy -> cuBLAS) and the accuracy against
  (1) the NumPy (CPU, OpenBLAS) result over the full matrix, and
  (2) a correctly rounded reference on randomly sampled entries
      (error-free TwoProduct + math.fsum), which tells whether the GPU result
      is as accurate as FP64 should be, independent of NumPy's own rounding.

Input matrices follow the convention of the Ozaki-scheme papers:
    A = (U(0,1) - 0.5) * exp(phi * N(0,1))
A larger phi widens the exponent range, which makes the emulation need more
slices/moduli (slower) to keep FP64 accuracy.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from common import (
    UNIT_ROUNDOFF,
    collect_env_info,
    condition_name,
    free_gpu_memory,
    log,
    summarize_times,
    time_gpu,
    write_json,
)

_SPLITTER = 134217729.0  # 2**27 + 1 (Veltkamp splitting for binary64)


def make_matrix(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Generate an n x n test matrix with a tunable exponent range."""
    a = rng.random((n, n)) - 0.5
    if phi != 0.0:
        a *= np.exp(phi * rng.standard_normal((n, n)))
    return a


def make_inputs(n: int, phi: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically generate (A, B) for a given size/phi/seed."""
    rng = np.random.default_rng([seed, n, int(round(phi * 1000))])
    return make_matrix(rng, n, phi), make_matrix(rng, n, phi)


def numpy_reference(a: np.ndarray, b: np.ndarray, cache_path: Path | None) -> tuple[np.ndarray, float | None]:
    """Compute (or load from cache) the NumPy FP64 product.

    Returns:
        The product and the CPU wall time in seconds (None if loaded from cache).
    """
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path), None
    t0 = time.perf_counter()
    c = a @ b
    elapsed = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.stem + ".tmp.npy")
        np.save(tmp, c)
        tmp.replace(cache_path)
    return c, elapsed


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = _SPLITTER * x
    hi = c - (c - x)
    return hi, x - hi


def exact_dot(x: np.ndarray, y: np.ndarray) -> float:
    """Correctly rounded dot product via Dekker's TwoProduct and math.fsum."""
    p = x * y
    xh, xl = _split(x)
    yh, yl = _split(y)
    e = ((xh * yh - p) + xh * yl + xl * yh) + xl * yl
    return math.fsum(np.concatenate([p, e]).tolist())


def sampled_errors(
    a: np.ndarray, b: np.ndarray, candidates: dict[str, np.ndarray], n_samples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Compare candidate products with a correctly rounded reference on sampled entries.

    The main metric is the componentwise error scaled by (|A||B|)_ij, expressed in
    units of the FP64 unit roundoff u = 2^-53. A classic FP64 dot product has a
    worst-case bound of about n*u in this metric, typically ~sqrt(n)*u.
    """
    n = a.shape[0]
    rng = np.random.default_rng([seed, n, 7])
    rows = rng.integers(0, n, n_samples)
    cols = rng.integers(0, n, n_samples)

    exact = np.empty(n_samples)
    scale = np.empty(n_samples)
    for k, (i, j) in enumerate(zip(rows, cols)):
        exact[k] = exact_dot(a[i, :], b[:, j])
        scale[k] = np.abs(a[i, :]) @ np.abs(b[:, j])

    out = {}
    for name, c in candidates.items():
        got = c[rows, cols]
        err = np.abs(got - exact)
        nonzero = exact != 0.0
        out[name] = {
            "max_scaled_err_in_u": float((err / scale).max() / UNIT_ROUNDOFF),
            "mean_scaled_err_in_u": float((err / scale).mean() / UNIT_ROUNDOFF),
            "max_elementwise_rel_err": float((err[nonzero] / np.abs(exact[nonzero])).max()),
            "median_elementwise_rel_err": float(np.median(err[nonzero] / np.abs(exact[nonzero]))),
        }
    return out


def full_errors(c: np.ndarray, c_ref: np.ndarray) -> dict[str, float]:
    """Normwise errors of c against c_ref over the full matrix."""
    diff = c - c_ref
    max_abs = float(np.abs(diff).max())
    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_abs / float(np.abs(c_ref).max()),
        "fro_rel_err": float(np.linalg.norm(diff) / np.linalg.norm(c_ref)),
    }


def gpu_matmul(a_h: np.ndarray, b_h: np.ndarray, warmup: int, repeats: int) -> tuple[list[float], np.ndarray]:
    """Time C = A @ B on the GPU (cuBLAS) and return (timings, C on host)."""
    import cupy as cp

    a_d = cp.asarray(a_h)
    b_d = cp.asarray(b_h)
    c_d = cp.empty(a_h.shape, dtype=cp.float64)
    times = time_gpu(lambda: cp.matmul(a_d, b_d, out=c_d), warmup, repeats)
    return times, cp.asnumpy(c_d)


def run_one(n: int, phi: float, args: argparse.Namespace) -> dict:
    """Benchmark one (n, phi) case."""
    flops = 2.0 * n**3
    record: dict = {"n": n, "phi": phi, "flops": flops}

    log(f"n={n} phi={phi}: generating inputs")
    a_h, b_h = make_inputs(n, phi, args.seed)

    log(f"n={n} phi={phi}: GPU timing ({args.warmup} warmup + {args.repeats} runs)")
    try:
        times, c_h = gpu_matmul(a_h, b_h, args.warmup, args.repeats)
    except Exception as exc:  # noqa: BLE001 - e.g. OOM at n=16000; keep the sweep going
        log(f"n={n} phi={phi}: FAILED ({exc!r})")
        record.update(status="failed", error=repr(exc))
        return record
    finally:
        free_gpu_memory()

    stats = summarize_times(times)
    record.update(
        status="ok",
        times_s=times,
        **stats,
        tflops_median=flops / stats["median_s"] / 1e12,
        tflops_best=flops / stats["min_s"] / 1e12,
    )
    log(f"n={n} phi={phi}: median {stats['median_s']:.4f} s, {record['tflops_median']:.3f} TFLOPS")

    candidates = {"gpu": c_h}
    if n <= args.cpu_ref_max_n:
        cache = None
        if not args.no_cache:
            cache = Path(args.cache_dir) / f"dgemm_ref_n{n}_phi{phi:g}_seed{args.seed}_np{np.__version__}.npy"
        log(f"n={n} phi={phi}: NumPy reference ({'cached' if cache and cache.exists() else 'computing'})")
        c_ref, cpu_time = numpy_reference(a_h, b_h, cache)
        record["cpu_time_s"] = cpu_time
        if cpu_time:
            record["cpu_tflops"] = flops / cpu_time / 1e12
        record["vs_numpy"] = full_errors(c_h, c_ref)
        candidates["numpy"] = c_ref
        log(f"n={n} phi={phi}: max rel err vs NumPy {record['vs_numpy']['max_rel_err']:.3e}")

    if args.samples > 0:
        log(f"n={n} phi={phi}: correctly rounded reference on {args.samples} sampled entries")
        record["vs_exact_sampled"] = sampled_errors(a_h, b_h, candidates, args.samples, args.seed)
        gpu_err = record["vs_exact_sampled"]["gpu"]["max_scaled_err_in_u"]
        log(f"n={n} phi={phi}: max scaled err (GPU) = {gpu_err:.2f} u")
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000, 16000])
    p.add_argument("--phi", type=float, nargs="+", default=[0.5], help="exponent-range parameter(s)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--samples", type=int, default=256, help="entries checked against a correctly rounded reference")
    p.add_argument("--cpu-ref-max-n", type=int, default=16000, help="skip the full NumPy reference above this size")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--no-cache", action="store_true", help="do not cache NumPy references on disk")
    p.add_argument("--output", default=None, help="default: results/<condition>/dgemm.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or f"results/{condition_name()}/dgemm.json"
    env = collect_env_info()
    log(f"CUDA {env['cuda_runtime']}, cuBLAS {env['cublas_version']}, {env['gpu_name']}, emu={env['emulation_env']}")

    payload = {"benchmark": "dgemm", "condition": condition_name(), "env": env, "config": vars(args), "results": []}
    for phi in args.phi:
        for n in args.sizes:
            payload["results"].append(run_one(n, phi, args))
            write_json(output, payload)  # checkpoint after every case
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
