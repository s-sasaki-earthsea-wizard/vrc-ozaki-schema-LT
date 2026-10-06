"""Minimal DGEMM driver to be run under Nsight Systems.

The timed iterations are wrapped in a single NVTX range named "dgemm", so that
`nsys stats --filter-nvtx dgemm` reports only the steady-state cuBLAS kernels
(no input transfer, no warm-up call). Inputs are generated exactly as in
benchmark_dgemm.py.
"""

from __future__ import annotations

import argparse

import cupy as cp
from cupy.cuda import nvtx

from benchmark_dgemm import make_inputs
from common import log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--phi", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261006)
    args = p.parse_args()

    a_h, b_h = make_inputs(args.n, args.phi, args.seed)
    a = cp.asarray(a_h)
    b = cp.asarray(b_h)
    c = cp.empty_like(a)

    cp.matmul(a, b, out=c)  # warm-up (handle creation, heuristics)
    cp.cuda.Device().synchronize()

    nvtx.RangePush("dgemm")
    for _ in range(args.iters):
        cp.matmul(a, b, out=c)
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    log(f"profiled n={args.n} phi={args.phi} iters={args.iters}")


if __name__ == "__main__":
    main()
