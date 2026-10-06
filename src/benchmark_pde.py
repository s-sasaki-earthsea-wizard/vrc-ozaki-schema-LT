"""2D heat equation u_t = alpha * (u_xx + u_yy) on [0,1]^2, Dirichlet u=0, FP64.

Two time integrators are benchmarked on an N x N interior grid (h = 1/(N+1)):

adi-gemm (Crank-Nicolson family, GEMM-bound)
    Peaceman-Rachford ADI, i.e. the factored Crank-Nicolson scheme. Because the
    1D operators in x and y commute on this grid, one step collapses to
        U <- M @ U @ M,   M = (I - a L)^-1 (I + a L),   a = alpha*dt/2,
    with L the 1D Dirichlet Laplacian. M is precomputed (dense), so each step is
    two N x N DGEMMs (4 N^3 flop). This is where cuBLAS FP64 emulation can help.

ftcs-stencil (explicit, memory-bound)
    Forward Euler + 5-point stencil in a custom CUDA kernel. It never calls cuBLAS,
    so FP64 emulation must NOT change its speed. It is the negative control that
    shows Ozaki is a GEMM accelerator, not a general FP64 accelerator.

Accuracy: the initial condition is a sum of discrete sine eigenmodes, so the
exact solution of the *discrete* scheme is known in closed form (each mode is
multiplied by its amplification factor per step). The error against it is pure
floating-point rounding. The error against the continuous solution
(discretization error) and against a NumPy (CPU) run are reported as well.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import collect_env_info, condition_name, free_gpu_memory, log, summarize_times, time_gpu, write_json

ALPHA = 1.0
# (kx, ky, coefficient) of the initial condition.
MODES = ((1, 1, 1.0), (2, 3, 0.5), (5, 4, 0.25), (7, 7, 0.1))

FTCS_KERNEL = r"""
extern "C" __global__
void ftcs_step(const double* __restrict__ u, double* __restrict__ v, const int n, const double r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > n || j > n) return;
    const int w = n + 2;
    const int k = i * w + j;
    v[k] = u[k] + r * (u[k - w] + u[k + w] + u[k - 1] + u[k + 1] - 4.0 * u[k]);
}
"""
FTCS_FLOPS_PER_POINT = 7.0
FTCS_BYTES_PER_POINT = 16.0  # one read + one write of FP64, assuming perfect reuse


# --------------------------------------------------------------------------- #
# Analytic pieces
# --------------------------------------------------------------------------- #
def grid_spacing(n: int) -> float:
    return 1.0 / (n + 1)


def sine_mode(n: int, k: int) -> np.ndarray:
    """Discrete eigenvector sin(k*pi*x_j), j = 1..n."""
    j = np.arange(1, n + 1)
    return np.sin(k * np.pi * j / (n + 1))


def laplacian_eigenvalue(n: int, k: int) -> float:
    """Eigenvalue of the 1D Dirichlet Laplacian tridiag(1,-2,1)/h^2 for mode k."""
    h = grid_spacing(n)
    return -4.0 / h**2 * np.sin(k * np.pi / (2 * (n + 1))) ** 2


def initial_condition(n: int) -> np.ndarray:
    u0 = np.zeros((n, n))
    for kx, ky, c in MODES:
        u0 += c * np.outer(sine_mode(n, kx), sine_mode(n, ky))
    return u0


def mode_superposition(n: int, factors: dict[tuple[int, int], float]) -> np.ndarray:
    """Sum of modes, each scaled by its coefficient times factors[(kx, ky)]."""
    u = np.zeros((n, n))
    for kx, ky, c in MODES:
        u += c * factors[(kx, ky)] * np.outer(sine_mode(n, kx), sine_mode(n, ky))
    return u


def adi_amplification(n: int, k: int, dt: float) -> float:
    a = ALPHA * dt / 2
    lam = laplacian_eigenvalue(n, k)
    return (1 + a * lam) / (1 - a * lam)


def ftcs_amplification(n: int, kx: int, ky: int, r: float) -> float:
    h2 = grid_spacing(n) ** 2
    return 1.0 + r * h2 * (laplacian_eigenvalue(n, kx) + laplacian_eigenvalue(n, ky))


def continuous_solution(n: int, t: float) -> np.ndarray:
    factors = {(kx, ky): np.exp(-ALPHA * np.pi**2 * (kx**2 + ky**2) * t) for kx, ky, _ in MODES}
    return mode_superposition(n, factors)


def adi_propagator(n: int, dt: float) -> np.ndarray:
    """Dense M = S diag(g) S with S the orthonormal DST-I matrix (symmetric)."""
    j = np.arange(1, n + 1)
    # Reduce j*k modulo the period 2(n+1) in exact integer arithmetic first; otherwise
    # the rounded argument (up to ~n rad) costs ~1e-12 accuracy at n = 8192.
    phase = np.outer(j, j) % (2 * (n + 1))
    s = np.sqrt(2.0 / (n + 1)) * np.sin(phase * (np.pi / (n + 1)))
    del phase
    g = np.array([adi_amplification(n, k, dt) for k in j])
    return (s * g) @ s


def error_metrics(u: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    max_abs = float(np.abs(u - ref).max())
    return {"max_abs_err": max_abs, "max_rel_err": max_abs / float(np.abs(ref).max())}


# --------------------------------------------------------------------------- #
# Schemes
# --------------------------------------------------------------------------- #
def run_adi_gpu(m_h: np.ndarray, u0_h: np.ndarray, steps: int, warmup: int, repeats: int):
    import cupy as cp

    m = cp.asarray(m_h)
    u0 = cp.asarray(u0_h)
    u = cp.empty_like(u0)
    tmp = cp.empty_like(u0)

    def integrate(n_steps: int) -> None:
        u[...] = u0
        for _ in range(n_steps):
            cp.matmul(m, u, out=tmp)
            cp.matmul(tmp, m, out=u)

    # Warm up separately so the timed run always starts from u0.
    time_gpu(lambda: integrate(1), warmup, 0)
    times = time_gpu(lambda: integrate(steps), 0, repeats)
    return times, cp.asnumpy(u)


def run_adi_cpu(m: np.ndarray, u0: np.ndarray, steps: int) -> np.ndarray:
    u = u0.copy()
    for _ in range(steps):
        u = (m @ u) @ m
    return u


def run_ftcs_gpu(u0_h: np.ndarray, steps: int, r: float, warmup: int, repeats: int):
    import cupy as cp

    n = u0_h.shape[0]
    kernel = cp.RawKernel(FTCS_KERNEL, "ftcs_step")
    block = (32, 8, 1)
    grid = ((n + block[0] - 1) // block[0], (n + block[1] - 1) // block[1], 1)

    padded0 = cp.zeros((n + 2, n + 2), dtype=cp.float64)
    padded0[1:-1, 1:-1] = cp.asarray(u0_h)
    bufs = [cp.empty_like(padded0), cp.zeros_like(padded0)]
    state = {"cur": 0}

    def integrate(n_steps: int) -> None:
        bufs[0][...] = padded0
        bufs[1].fill(0.0)
        cur = 0
        for _ in range(n_steps):
            kernel(grid, block, (bufs[cur], bufs[1 - cur], np.int32(n), np.float64(r)))
            cur = 1 - cur
        state["cur"] = cur

    time_gpu(lambda: integrate(1), warmup, 0)
    times = time_gpu(lambda: integrate(steps), 0, repeats)
    return times, cp.asnumpy(bufs[state["cur"]][1:-1, 1:-1])


def run_ftcs_cpu(u0: np.ndarray, steps: int, r: float) -> np.ndarray:
    n = u0.shape[0]
    u = np.zeros((n + 2, n + 2))
    u[1:-1, 1:-1] = u0
    v = np.zeros_like(u)
    for _ in range(steps):
        c = u[1:-1, 1:-1]
        v[1:-1, 1:-1] = c + r * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * c)
        u, v = v, u
    return u[1:-1, 1:-1].copy()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def cached_cpu_run(cache: Path | None, compute) -> tuple[np.ndarray, float | None]:
    if cache is not None and cache.exists():
        return np.load(cache), None
    t0 = time.perf_counter()
    u = compute()
    elapsed = time.perf_counter() - t0
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npy")
        np.save(tmp, u)
        tmp.replace(cache)
    return u, elapsed


def adi_exact(n: int, dt: float, steps: int) -> np.ndarray:
    """Exact solution of the discrete ADI scheme after `steps` steps."""
    g = {k: adi_amplification(n, k, dt) for k in {mm for kx, ky, _ in MODES for mm in (kx, ky)}}
    return mode_superposition(n, {(kx, ky): (g[kx] * g[ky]) ** steps for kx, ky, _ in MODES})


def ftcs_exact(n: int, r: float, steps: int) -> np.ndarray:
    """Exact solution of the discrete FTCS scheme after `steps` steps."""
    return mode_superposition(n, {(kx, ky): ftcs_amplification(n, kx, ky, r) ** steps for kx, ky, _ in MODES})


def numpy_ref_cache_path(cache_dir: str, scheme: str, n: int, steps: int, dt: float) -> Path:
    """Cache file of the NumPy reference run, shared by the GPU and CPU benchmarks."""
    return Path(cache_dir) / f"pde_ref_{scheme}_n{n}_steps{steps}_{dt:.6g}_np{np.__version__}.npy"


def run_case(scheme: str, n: int, args: argparse.Namespace) -> dict:
    u0 = initial_condition(n)
    h = grid_spacing(n)

    if scheme == "adi-gemm":
        steps, dt = args.adi_steps, args.adi_dt
        flops_per_step = 4.0 * n**3
        log(f"{scheme} n={n}: building propagator on CPU")
        m = adi_propagator(n, dt)
        exact = adi_exact(n, dt, steps)
        gpu_run = lambda: run_adi_gpu(m, u0, steps, args.warmup, args.repeats)  # noqa: E731
        cpu_run = lambda: run_adi_cpu(m, u0, steps)  # noqa: E731
        extra = {"dt": dt, "courant_number_r": ALPHA * dt / h**2}
    elif scheme == "ftcs-stencil":
        steps, r = args.ftcs_steps, args.ftcs_r
        dt = r * h**2 / ALPHA
        flops_per_step = FTCS_FLOPS_PER_POINT * n**2
        exact = ftcs_exact(n, r, steps)
        gpu_run = lambda: run_ftcs_gpu(u0, steps, r, args.warmup, args.repeats)  # noqa: E731
        cpu_run = lambda: run_ftcs_cpu(u0, steps, r)  # noqa: E731
        extra = {"dt": dt, "courant_number_r": r}
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    record: dict = {"scheme": scheme, "n": n, "steps": steps, "flops_per_step": flops_per_step, **extra}

    log(f"{scheme} n={n}: GPU timing ({steps} steps x {args.repeats} runs)")
    try:
        times, u_gpu = gpu_run()
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        log(f"{scheme} n={n}: FAILED ({exc!r})")
        record.update(status="failed", error=repr(exc))
        return record
    finally:
        free_gpu_memory()

    stats = summarize_times(times)
    per_step = stats["median_s"] / steps
    record.update(
        status="ok",
        times_s=times,
        **stats,
        time_per_step_s=per_step,
        tflops=flops_per_step / per_step / 1e12,
    )
    if scheme == "ftcs-stencil":
        record["effective_bandwidth_gbs"] = FTCS_BYTES_PER_POINT * n**2 / per_step / 1e9
    log(f"{scheme} n={n}: {per_step * 1e3:.3f} ms/step, {record['tflops']:.4f} TFLOPS")

    record["vs_discrete_exact"] = error_metrics(u_gpu, exact)
    record["discretization_err_vs_continuous"] = error_metrics(exact, continuous_solution(n, steps * dt))

    if n <= args.cpu_ref_max_n:
        cache = None if args.no_cache else numpy_ref_cache_path(args.cache_dir, scheme, n, steps, extra["dt"])
        log(f"{scheme} n={n}: NumPy reference ({'cached' if cache and cache.exists() else 'computing'})")
        u_cpu, cpu_time = cached_cpu_run(cache, cpu_run)
        record["cpu_time_s"] = cpu_time
        record["vs_numpy"] = error_metrics(u_gpu, u_cpu)
        record["numpy_vs_discrete_exact"] = error_metrics(u_cpu, exact)

    log(f"{scheme} n={n}: rel err vs discrete exact {record['vs_discrete_exact']['max_rel_err']:.3e}")
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--schemes", nargs="+", default=["adi-gemm", "ftcs-stencil"], choices=["adi-gemm", "ftcs-stencil"])
    p.add_argument("--adi-steps", type=int, default=20)
    p.add_argument("--adi-dt", type=float, default=1e-4)
    p.add_argument("--ftcs-steps", type=int, default=1000)
    p.add_argument("--ftcs-r", type=float, default=0.2, help="alpha*dt/h^2 (stable for <= 0.25)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-ref-max-n", type=int, default=2048)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output", default=None, help="default: results/<condition>/pde.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ftcs_r > 0.25:
        raise SystemExit(f"--ftcs-r {args.ftcs_r} violates the FTCS stability limit 0.25")
    output = args.output or f"results/{condition_name()}/pde.json"
    env = collect_env_info()
    log(f"CUDA {env['cuda_runtime']}, cuBLAS {env['cublas_version']}, {env['gpu_name']}, emu={env['emulation_env']}")

    payload = {"benchmark": "pde", "condition": condition_name(), "env": env, "config": vars(args), "results": []}
    for scheme in args.schemes:
        for n in args.sizes:
            payload["results"].append(run_case(scheme, n, args))
            write_json(output, payload)
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
