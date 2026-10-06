"""CPU FP64 baselines: C++/Eigen (cpp/cpu_bench) and NumPy (OpenBLAS), with a thread sweep.

Inputs are generated exactly as in the GPU benchmarks (same seeds, same functions),
handed to the C++ binary as .npy files, and the results are evaluated with the same
accuracy metrics. Output files follow the GPU schema, one record per thread count:

    results/cpu-eigen/{env,dgemm,pde}.json
    results/cpu-numpy/{env,dgemm}.json

Usage:
    python src/benchmark_cpu.py dgemm --sizes 2000 4000 8000 16000 --threads 6 14 16
    python src/benchmark_cpu.py pde   --sizes 1024 2048 4096 8192  --threads 6 14 16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import benchmark_dgemm as dg
import benchmark_pde as pde
from common import log, summarize_times, write_json

EIGEN = "cpu-eigen"
NUMPY = "cpu-numpy"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def collect_cpu_env_info(binary: str, condition: str) -> dict:
    """CPU-side counterpart of common.collect_env_info."""
    from threadpoolctl import threadpool_info

    try:
        cpp = json.loads(subprocess.run([binary, "version"], capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        cpp = {"error": repr(exc)}
    return {
        "condition": condition,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threadpools": [
            {k: p.get(k) for k in ("user_api", "internal_api", "version", "num_threads", "architecture")}
            for p in threadpool_info()
        ],
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpp": cpp,
        "omp_env": {k: os.environ.get(k) for k in ("OMP_PLACES", "OMP_PROC_BIND")},
    }


def run_binary(binary: str, command: str, **kwargs) -> dict:
    """Run cpu_bench and return its JSON report."""
    cmd = [binary, command]
    for key, value in kwargs.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def thread_list(threads: list[int]) -> str:
    return ",".join(str(t) for t in threads)


def timing_fields(times: list[float], flops: float) -> dict:
    stats = summarize_times(times)
    return {
        "status": "ok",
        "times_s": times,
        **stats,
        "tflops_median": flops / stats["median_s"] / 1e12,
        "tflops_best": flops / stats["min_s"] / 1e12,
    }


# --------------------------------------------------------------------------- #
# DGEMM
# --------------------------------------------------------------------------- #
def time_numpy_dgemm(a: np.ndarray, b: np.ndarray, threads: int, repeats: int) -> tuple[list[float], np.ndarray]:
    from threadpoolctl import threadpool_limits

    c = np.empty((a.shape[0], b.shape[1]))
    times = []
    with threadpool_limits(limits=threads, user_api="blas"):
        np.matmul(a[:256, :256], b[:256, :256])  # spin up the BLAS thread pool
        for _ in range(repeats):
            t0 = time.perf_counter()
            np.matmul(a, b, out=c)
            times.append(time.perf_counter() - t0)
    return times, c


def run_dgemm(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = {
        EIGEN: {"benchmark": "dgemm", "condition": EIGEN, "config": vars(args), "results": []},
        NUMPY: {"benchmark": "dgemm", "condition": NUMPY, "config": vars(args), "results": []},
    }
    for cond in out:
        out[cond]["env"] = collect_cpu_env_info(args.binary, cond)
        write_json(Path(args.results_dir) / cond / "env.json", out[cond]["env"])
    paths = {cond: Path(args.results_dir) / cond / "dgemm.json" for cond in out}
    run_numpy = not args.no_numpy

    for phi in args.phi:
        for n in args.sizes:
            flops = 2.0 * n**3
            log(f"dgemm n={n} phi={phi}: generating inputs")
            a, b = dg.make_inputs(n, phi, args.seed)
            np.save(work / "a.npy", a)
            np.save(work / "b.npy", b)

            log(f"dgemm n={n} phi={phi}: Eigen, threads {args.threads}")
            report = run_binary(
                args.binary, "dgemm", a=work / "a.npy", b=work / "b.npy", out=work / "c.npy",
                threads=thread_list(args.threads), repeats=args.repeats,
            )
            c_eigen = np.load(work / "c.npy")

            cache = None if args.no_cache else (
                Path(args.cache_dir) / f"dgemm_ref_n{n}_phi{phi:g}_seed{args.seed}_np{np.__version__}.npy"
            )
            c_ref, _ = dg.numpy_reference(a, b, cache)

            numpy_runs = []
            if run_numpy:
                for t in args.threads:
                    log(f"dgemm n={n} phi={phi}: NumPy, {t} threads")
                    times, c_np = time_numpy_dgemm(a, b, t, args.repeats)
                    # OpenBLAS rounding can depend on the thread count, so evaluate each run's own result.
                    own = dg.sampled_errors(a, b, {"result": c_np}, args.samples, args.seed)["result"] \
                        if args.samples > 0 else None
                    numpy_runs.append((t, times, float(np.abs(c_np - c_ref).max()), own))
                    del c_np

            sampled = dg.sampled_errors(a, b, {"eigen": c_eigen, "numpy": c_ref}, args.samples, args.seed) \
                if args.samples > 0 else {}
            vs_numpy = dg.full_errors(c_eigen, c_ref)

            for run in report["runs"]:
                rec = {"n": n, "phi": phi, "flops": flops, "threads": run["threads"],
                       **timing_fields(run["times_s"], flops), "vs_numpy": vs_numpy,
                       "max_abs_diff_between_thread_configs": report["max_abs_diff_between_thread_configs"]}
                if sampled:
                    rec["vs_exact_sampled"] = {"result": sampled["eigen"], "numpy": sampled["numpy"]}
                out[EIGEN]["results"].append(rec)
                log(f"eigen n={n} t={run['threads']}: {rec['median_s']:.3f} s, {rec['tflops_median']:.3f} TFLOPS")

            for t, times, diff, own in numpy_runs:
                rec = {"n": n, "phi": phi, "flops": flops, "threads": t, **timing_fields(times, flops),
                       "vs_numpy": {"max_abs_err": diff, "max_rel_err": diff / float(np.abs(c_ref).max())}}
                if own is not None:
                    rec["vs_exact_sampled"] = {"result": own, "numpy": sampled["numpy"]}
                out[NUMPY]["results"].append(rec)
                log(f"numpy n={n} t={t}: {rec['median_s']:.3f} s, {rec['tflops_median']:.3f} TFLOPS")

            for cond, path in paths.items():
                if cond == EIGEN or run_numpy:
                    write_json(path, out[cond])
            del a, b, c_eigen, c_ref


# --------------------------------------------------------------------------- #
# PDE
# --------------------------------------------------------------------------- #
def run_pde(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": "pde", "condition": EIGEN, "env": collect_cpu_env_info(args.binary, EIGEN),
               "config": vars(args), "results": []}
    path = Path(args.results_dir) / EIGEN / "pde.json"
    write_json(path.with_name("env.json"), payload["env"])

    for scheme in args.schemes:
        for n in args.sizes:
            h = pde.grid_spacing(n)
            u0 = pde.initial_condition(n)
            np.save(work / "u0.npy", u0)
            if scheme == "adi-gemm":
                steps, dt = args.adi_steps, args.adi_dt
                flops_per_step = 4.0 * n**3
                log(f"{scheme} n={n}: building propagator")
                m = pde.adi_propagator(n, dt)
                np.save(work / "m.npy", m)
                exact = pde.adi_exact(n, dt, steps)
                extra = {"dt": dt, "courant_number_r": pde.ALPHA * dt / h**2}
                kwargs = {"m": work / "m.npy"}
                numpy_run = lambda: pde.run_adi_cpu(m, u0, steps)  # noqa: E731
                command = "adi"
            else:
                steps, r = args.ftcs_steps, args.ftcs_r
                dt = r * h**2 / pde.ALPHA
                flops_per_step = pde.FTCS_FLOPS_PER_POINT * n**2
                exact = pde.ftcs_exact(n, r, steps)
                extra = {"dt": dt, "courant_number_r": r}
                kwargs = {"r": repr(r)}
                numpy_run = lambda: pde.run_ftcs_cpu(u0, steps, r)  # noqa: E731
                command = "ftcs"

            log(f"{scheme} n={n}: Eigen/C++, threads {args.threads}, {steps} steps")
            report = run_binary(
                args.binary, command, u0=work / "u0.npy", out=work / "u.npy", steps=steps,
                threads=thread_list(args.threads), repeats=args.repeats, **kwargs,
            )
            u = np.load(work / "u.npy")

            errors = {
                "vs_discrete_exact": pde.error_metrics(u, exact),
                "discretization_err_vs_continuous": pde.error_metrics(exact, pde.continuous_solution(n, steps * dt)),
            }
            if n <= args.cpu_ref_max_n:
                cache = None if args.no_cache else pde.numpy_ref_cache_path(args.cache_dir, scheme, n, steps, dt)
                u_np, _ = pde.cached_cpu_run(cache, numpy_run)
                errors["vs_numpy"] = pde.error_metrics(u, u_np)
                errors["numpy_vs_discrete_exact"] = pde.error_metrics(u_np, exact)

            for run in report["runs"]:
                stats = summarize_times(run["times_s"])
                per_step = stats["median_s"] / steps
                rec = {"scheme": scheme, "n": n, "steps": steps, "threads": run["threads"],
                       "flops_per_step": flops_per_step, **extra, "status": "ok", "times_s": run["times_s"], **stats,
                       "time_per_step_s": per_step, "tflops": flops_per_step / per_step / 1e12, **errors,
                       "max_abs_diff_between_thread_configs": report["max_abs_diff_between_thread_configs"]}
                if scheme == "ftcs-stencil":
                    rec["effective_bandwidth_gbs"] = pde.FTCS_BYTES_PER_POINT * n**2 / per_step / 1e9
                payload["results"].append(rec)
                log(f"{scheme} n={n} t={run['threads']}: {per_step * 1e3:.3f} ms/step, {rec['tflops']:.4f} TFLOPS")
            write_json(path, payload)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--threads", type=int, nargs="+", default=[6, 14, 16])
        sp.add_argument("--binary", default="build/cpp/cpu_bench")
        sp.add_argument("--results-dir", default="results")
        sp.add_argument("--work-dir", default="cache/cpu_work", help="scratch space for .npy hand-off")
        sp.add_argument("--cache-dir", default="cache")
        sp.add_argument("--no-cache", action="store_true")

    d = sub.add_parser("dgemm")
    common(d)
    d.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000, 16000])
    d.add_argument("--phi", type=float, nargs="+", default=[0.5])
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--seed", type=int, default=20261006)
    d.add_argument("--samples", type=int, default=256)
    d.add_argument("--no-numpy", action="store_true", help="skip the NumPy (OpenBLAS) thread sweep")

    q = sub.add_parser("pde")
    common(q)
    q.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    q.add_argument("--schemes", nargs="+", default=["adi-gemm", "ftcs-stencil"], choices=["adi-gemm", "ftcs-stencil"])
    q.add_argument("--adi-steps", type=int, default=20)
    q.add_argument("--adi-dt", type=float, default=1e-4)
    q.add_argument("--ftcs-steps", type=int, default=1000)
    q.add_argument("--ftcs-r", type=float, default=0.2)
    q.add_argument("--repeats", type=int, default=1)
    q.add_argument("--cpu-ref-max-n", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("BENCH_CONDITION", "cpu")
    if args.command == "dgemm":
        run_dgemm(args)
    else:
        run_pde(args)
    log("done")


if __name__ == "__main__":
    main()
