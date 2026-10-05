"""Summarize Nsight Systems kernel reports produced by profile_kernels.sh.

Reads results/nsys/<condition>/n<N>_phi<phi>_cuda_gpu_kern_sum_nvtx=dgemm.csv and
classifies every kernel to decide which DGEMM path cuBLAS actually took:
Ozaki-II (kernel names contain "oz2"), Ozaki-I, or native FP64.
Writes kernels.md and kernels.json next to the CSV directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

CSV_PATTERN = re.compile(r"n(?P<n>\d+)_phi(?P<phi>[\d.]+)_cuda_gpu_kern_sum.*\.csv$")
CONDITION_ORDER = ["cuda12", "cuda13-native", "cuda13-emu", "cuda13-emu-eager"]


def classify(name: str) -> str:
    """Map a kernel name to a coarse category."""
    s = name.lower()
    if "oz2" in s:
        return "ozaki2"
    if "oz1" in s or "ozaki1" in s or "ozimmu" in s:
        return "ozaki1"
    if "_emu" in s:
        return "emulation-other"
    if ("nvjet" in s and ("biu" in s or "i8" in s)) or "imma" in s or "igemm" in s:
        return "int8-mma"
    if "d884gemm" in s or "dgemm" in s or "f64" in s or "dmma" in s:
        return "native-fp64"
    return "other"


def verdict(share: dict[str, float]) -> str:
    """Decide which DGEMM path was taken from the time share per category."""
    if share.get("ozaki2", 0) > 0:
        return "Ozaki-II"
    if share.get("ozaki1", 0) > 0:
        return "Ozaki-I"
    if share.get("int8-mma", 0) + share.get("emulation-other", 0) > 0:
        return "emulated (scheme unknown)"
    if share.get("native-fp64", 0) > 0:
        return "native FP64"
    return "unknown"


def short_name(name: str, width: int = 70) -> str:
    """Strip 'void ', template arguments and parameter lists for display."""
    base = name.removeprefix("void ")
    head = base.split("(")[0]
    if "<" in head:
        outer = head.split("<", 1)
        first_arg = outer[1].split(",")[0].rstrip(">")
        head = f"{outer[0]}<{first_arg},...>"
    return head if len(head) <= width else head[: width - 3] + "..."


def load_case(path: Path, iters: int) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total_ns = sum(float(r["Total Time (ns)"]) for r in rows)
    by_cat: dict[str, float] = defaultdict(float)
    for r in rows:
        by_cat[classify(r["Name"])] += float(r["Total Time (ns)"])
    share = {k: v / total_ns for k, v in by_cat.items()} if total_ns else {}
    top = sorted(rows, key=lambda r: float(r["Total Time (ns)"]), reverse=True)[:4]
    m = CSV_PATTERN.search(path.name)
    return {
        "condition": path.parent.name,
        "n": int(m["n"]),
        "phi": float(m["phi"]),
        "gpu_time_per_dgemm_ms": total_ns / iters / 1e6,
        "path": verdict(share),
        "share": share,
        # Proxy for the number of moduli: the INT8 GEMMs are batched over moduli.
        "int8_mma_ms_per_dgemm": by_cat.get("int8-mma", 0.0) / iters / 1e6,
        "kernel_count": len(rows),
        "top_kernels": [
            {"name": r["Name"], "short": short_name(r["Name"]), "time_pct": float(r["Time (%)"])} for r in top
        ],
    }


def to_markdown(cases: list[dict]) -> str:
    lines = [
        "# DGEMM kernels observed with Nsight Systems",
        "",
        "Kernels inside the NVTX range `dgemm` (steady state, excluding warm-up and transfers).",
        "*path* is derived from kernel names: `oz2_*` = Ozaki-II, `d884gemm` = native FP64 (DMMA).",
        "*INT8 MMA ms* = time in INT8 tensor-core GEMMs per DGEMM; these are batched over the moduli,",
        "so at fixed N it grows with the number of moduli that cuBLAS selects.",
        "",
        "| condition | N | phi | path | GPU ms / DGEMM | INT8 MMA ms | INT8 MMA share | top kernels (time %) |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for c in cases:
        top = "<br>".join(f"`{k['short']}` ({k['time_pct']:.1f}%)" for k in c["top_kernels"])
        int8 = c["share"].get("int8-mma", 0.0)
        lines.append(
            f"| {c['condition']} | {c['n']} | {c['phi']:g} | {c['path']} | {c['gpu_time_per_dgemm_ms']:.3f} "
            f"| {c['int8_mma_ms_per_dgemm']:.3f} | {int8:.0%} | {top} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("results/nsys"))
    p.add_argument("--iters", type=int, default=3, help="DGEMM calls inside the NVTX range")
    args = p.parse_args()

    rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    cases = [load_case(f, args.iters) for f in args.dir.glob("*/*.csv") if CSV_PATTERN.search(f.name)]
    if not cases:
        raise SystemExit(f"no kernel summaries found under {args.dir}")
    cases.sort(key=lambda c: (c["phi"], c["n"], rank.get(c["condition"], 99)))

    (args.dir / "kernels.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (args.dir / "kernels.md").write_text(to_markdown(cases), encoding="utf-8")
    print(f"wrote {args.dir / 'kernels.md'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
