"""Aggregate results/<condition>/{dgemm,pde}.json into PNG figures and a Markdown report.

Usage:
    python src/analyze_results.py --results-dir results \
        --conditions cuda12 cuda13-native cuda13-emu cuda13-emu-eager
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated categorical palette (fixed order, never cycled) + light chart surface.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE_GRAY = "#8a8984"

PREFERRED_ORDER = ["cuda12", "cuda12.4", "cuda13-native", "cuda13-emu", "cuda13-emu-eager"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def discover_conditions(results_dir: Path) -> list[str]:
    found = [p.name for p in results_dir.iterdir() if p.is_dir() and any(p.glob("*.json"))]
    rank = {c: i for i, c in enumerate(PREFERRED_ORDER)}
    return sorted(found, key=lambda c: (rank.get(c, len(rank)), c))


def load(results_dir: Path, conditions: list[str], bench: str) -> dict[str, dict]:
    out = {}
    for cond in conditions:
        path = results_dir / cond / f"{bench}.json"
        if path.exists():
            out[cond] = json.loads(path.read_text(encoding="utf-8"))
    return out


def ok_rows(payload: dict) -> list[dict]:
    return [r for r in payload["results"] if r.get("status") == "ok"]


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def new_figure(ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.4), facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        style_axes(ax)
    return fig, axes[0]


def finish(fig, path: Path, title: str) -> None:
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def series_style(conditions: list[str], cond: str) -> dict:
    i = conditions.index(cond)
    return {
        "color": SERIES_COLORS[i % len(SERIES_COLORS)],
        "marker": SERIES_MARKERS[i % len(SERIES_MARKERS)],
        "linewidth": 2,
        "markersize": 7,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 1.5,
        "label": cond,
    }


def legend(ax) -> None:
    leg = ax.legend(frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_PRIMARY)


def set_size_ticks(ax, sizes: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()


def speedup_axis(ax) -> None:
    """Linear y axis from 0 with 'Nx' tick labels (easier to read on a slide than log)."""
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))


# --------------------------------------------------------------------------- #
# DGEMM
# --------------------------------------------------------------------------- #
def dgemm_lookup(data: dict[str, dict]) -> dict[tuple[str, float, int], dict]:
    return {(c, r["phi"], r["n"]): r for c, p in data.items() for r in ok_rows(p)}


def plot_dgemm(data: dict[str, dict], conditions: list[str], baseline: str, fig_dir: Path) -> list[str]:
    lut = dgemm_lookup(data)
    phis = sorted({k[1] for k in lut})
    sizes = sorted({k[2] for k in lut})
    figs = []

    # Throughput
    fig, axes = new_figure(len(phis))
    for ax, phi in zip(axes, phis):
        for cond in conditions:
            pts = [(n, lut[(cond, phi, n)]["tflops_median"]) for n in sizes if (cond, phi, n) in lut]
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        set_size_ticks(ax, sizes)
        ax.set_yscale("log")
        ax.set_xlabel("matrix size N")
        ax.set_ylabel("TFLOPS (median)")
        ax.set_title(f"phi = {phi:g}")
        legend(ax)
    finish(fig, fig_dir / "dgemm_tflops.png", "FP64 DGEMM throughput")
    figs.append("dgemm_tflops.png")

    # Speedup vs baseline
    others = [c for c in conditions if c != baseline]
    if baseline in data and others:
        fig, axes = new_figure(len(phis))
        for ax, phi in zip(axes, phis):
            for cond in others:
                pts = [
                    (n, lut[(baseline, phi, n)]["median_s"] / lut[(cond, phi, n)]["median_s"])
                    for n in sizes
                    if (cond, phi, n) in lut and (baseline, phi, n) in lut
                ]
                if pts:
                    ax.plot(*zip(*pts), **series_style(conditions, cond))
            ax.axhline(1.0, color=REFERENCE_GRAY, linewidth=1, linestyle="--")
            set_size_ticks(ax, sizes)
            speedup_axis(ax)
            ax.set_xlabel("matrix size N")
            ax.set_ylabel(f"speedup vs {baseline}")
            ax.set_title(f"phi = {phi:g}")
            legend(ax)
        finish(fig, fig_dir / "dgemm_speedup.png", f"DGEMM speedup relative to {baseline}")
        figs.append("dgemm_speedup.png")

    # Accuracy against the correctly rounded reference
    fig, axes = new_figure(len(phis))
    for ax, phi in zip(axes, phis):
        numpy_ref: dict[int, float] = {}  # identical inputs in every condition, so keep the first
        for cond in conditions:
            pts = []
            for n in sizes:
                ex = lut.get((cond, phi, n), {}).get("vs_exact_sampled")
                if ex:
                    pts.append((n, ex["gpu"]["max_scaled_err_in_u"]))
                    if "numpy" in ex:
                        numpy_ref.setdefault(n, ex["numpy"]["max_scaled_err_in_u"])
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        if numpy_ref:
            ax.plot(*zip(*sorted(numpy_ref.items())), color=REFERENCE_GRAY, linestyle="--", linewidth=1.5,
                    label="NumPy (CPU)")
        set_size_ticks(ax, sizes)
        ax.set_yscale("log")
        ax.set_xlabel("matrix size N")
        ax.set_ylabel("max |C - C_exact| / (|A||B|)  [units of u = 2^-53]")
        ax.set_title(f"phi = {phi:g}")
        legend(ax)
    finish(fig, fig_dir / "dgemm_error.png", "DGEMM accuracy vs correctly rounded reference (sampled)")
    figs.append("dgemm_error.png")
    return figs


def dgemm_table(data: dict[str, dict], conditions: list[str], baseline: str, native13: str) -> str:
    lut = dgemm_lookup(data)
    lines = [
        "| phi | N | condition | median [s] | TFLOPS | speedup vs "
        f"{baseline} | speedup vs {native13} | rel err vs NumPy | max scaled err [u] (GPU / NumPy) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(k[1], k[2]) for k in lut})
    for phi, n in keys:
        for cond in conditions:
            r = lut.get((cond, phi, n))
            if r is None:
                failed = [x for x in data.get(cond, {}).get("results", []) if x["n"] == n and x["phi"] == phi]
                if failed:
                    lines.append(f"| {phi:g} | {n} | {cond} | failed | | | | | `{failed[0].get('error', '')[:60]}` |")
                continue
            sp_base = _ratio(lut.get((baseline, phi, n)), r)
            sp_nat = _ratio(lut.get((native13, phi, n)), r)
            rel = r.get("vs_numpy", {}).get("max_rel_err")
            ex = r.get("vs_exact_sampled", {})
            gpu_u = ex.get("gpu", {}).get("max_scaled_err_in_u")
            np_u = ex.get("numpy", {}).get("max_scaled_err_in_u")
            lines.append(
                f"| {phi:g} | {n} | {cond} | {r['median_s']:.4f} | {r['tflops_median']:.3f} | {sp_base} | {sp_nat} "
                f"| {_sci(rel)} | {_num(gpu_u)} / {_num(np_u)} |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PDE
# --------------------------------------------------------------------------- #
def pde_lookup(data: dict[str, dict]) -> dict[tuple[str, str, int], dict]:
    return {(c, r["scheme"], r["n"]): r for c, p in data.items() for r in ok_rows(p)}


def plot_pde(data: dict[str, dict], conditions: list[str], baseline: str, fig_dir: Path) -> list[str]:
    lut = pde_lookup(data)
    schemes = sorted({k[1] for k in lut})
    sizes = sorted({k[2] for k in lut})
    figs = []

    fig, axes = new_figure(len(schemes))
    for ax, scheme in zip(axes, schemes):
        for cond in conditions:
            pts = [(n, lut[(cond, scheme, n)]["time_per_step_s"] * 1e3) for n in sizes if (cond, scheme, n) in lut]
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        set_size_ticks(ax, sizes)
        ax.set_yscale("log")
        ax.set_xlabel("grid size N (N x N)")
        ax.set_ylabel("time per step [ms] (median)")
        ax.set_title(scheme)
        legend(ax)
    finish(fig, fig_dir / "pde_time_per_step.png", "2D heat equation: time per step")
    figs.append("pde_time_per_step.png")

    others = [c for c in conditions if c != baseline]
    if baseline in data and others:
        fig, axes = new_figure(len(schemes))
        for ax, scheme in zip(axes, schemes):
            for cond in others:
                pts = [
                    (n, lut[(baseline, scheme, n)]["time_per_step_s"] / lut[(cond, scheme, n)]["time_per_step_s"])
                    for n in sizes
                    if (cond, scheme, n) in lut and (baseline, scheme, n) in lut
                ]
                if pts:
                    ax.plot(*zip(*pts), **series_style(conditions, cond))
            ax.axhline(1.0, color=REFERENCE_GRAY, linewidth=1, linestyle="--")
            set_size_ticks(ax, sizes)
            speedup_axis(ax)
            ax.set_xlabel("grid size N (N x N)")
            ax.set_ylabel(f"speedup vs {baseline}")
            ax.set_title(scheme)
            legend(ax)
        finish(fig, fig_dir / "pde_speedup.png", f"2D heat equation: speedup relative to {baseline}")
        figs.append("pde_speedup.png")
    return figs


def pde_table(data: dict[str, dict], conditions: list[str], baseline: str, native13: str) -> str:
    lut = pde_lookup(data)
    lines = [
        f"| scheme | N | steps | condition | ms/step | TFLOPS | speedup vs {baseline} | speedup vs {native13} "
        "| rel err vs discrete exact | rel err vs NumPy | discretization err |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme, n in sorted({(k[1], k[2]) for k in lut}):
        for cond in conditions:
            r = lut.get((cond, scheme, n))
            if r is None:
                continue
            base = lut.get((baseline, scheme, n))
            nat = lut.get((native13, scheme, n))
            sp_base = f"{base['time_per_step_s'] / r['time_per_step_s']:.2f}x" if base else "-"
            sp_nat = f"{nat['time_per_step_s'] / r['time_per_step_s']:.2f}x" if nat else "-"
            lines.append(
                f"| {scheme} | {n} | {r['steps']} | {cond} | {r['time_per_step_s'] * 1e3:.3f} | {r['tflops']:.4f} "
                f"| {sp_base} | {sp_nat} | {_sci(r['vs_discrete_exact']['max_rel_err'])} "
                f"| {_sci(r.get('vs_numpy', {}).get('max_rel_err'))} "
                f"| {_sci(r['discretization_err_vs_continuous']['max_rel_err'])} |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _ratio(ref: dict | None, r: dict) -> str:
    return f"{ref['median_s'] / r['median_s']:.2f}x" if ref else "-"


def _sci(v: float | None) -> str:
    return "-" if v is None else f"{v:.2e}"


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def env_table(payloads: dict[str, dict], conditions: list[str]) -> str:
    lines = [
        "| condition | GPU (cc) | kernel driver | libcuda (API) | CUDA toolkit | cuBLAS | CuPy | NumPy "
        "| emulation env | libcublas |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond in conditions:
        if cond not in payloads:
            continue
        e = payloads[cond]["env"]
        emu = ", ".join(f"{k.removeprefix('CUBLAS_')}={v}" for k, v in e["emulation_env"].items() if v is not None)
        libs = "<br>".join(e.get("libcublas_loaded", [])) or "-"
        libcuda = ", ".join(Path(p).name for p in e.get("libcuda_loaded", [])) or "-"
        lines.append(
            f"| {cond} | {e['gpu_name']} ({e['compute_capability']}) | {e['nvidia_smi'].get('driver_version', '-')} "
            f"| {libcuda} ({e['cuda_driver_api']}) | {e.get('cuda_toolkit') or '-'} | {e['cublas_version']} "
            f"| {e['cupy']} | {e['numpy']} | {emu or '(unset)'} | {libs} |"
        )
    return "\n".join(lines)


def write_report(args, conditions, dgemm, pde, figs) -> Path:
    any_payload = {**pde, **dgemm}
    parts = [
        "# CUDA 12 vs CUDA 13.4 (Ozaki scheme) FP64 benchmark report",
        "",
        f"- conditions: {', '.join(conditions)}",
        f"- baseline for speedup: `{args.baseline}`; emulation effect isolated against `{args.native13}`",
        "",
        "## Environment",
        "",
        env_table(any_payload, conditions),
        "",
        "Speedups are ratios of median times. Errors: *rel err vs NumPy* = max|C-C_np| / max|C_np|;",
        "*scaled err* = max over sampled entries of |c_ij - exact_ij| / (|A||B|)_ij in units of u = 2^-53.",
        "",
    ]
    if dgemm:
        parts += ["## DGEMM", ""]
        parts += [f"![{f}](figures/{f})" for f in figs if f.startswith("dgemm")]
        parts += ["", dgemm_table(dgemm, conditions, args.baseline, args.native13), ""]
    if pde:
        parts += ["## 2D heat equation", ""]
        parts += [f"![{f}](figures/{f})" for f in figs if f.startswith("pde")]
        parts += ["", pde_table(pde, conditions, args.baseline, args.native13), ""]
    out = args.results_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--conditions", nargs="+", default=None, help="order defines colors; default: auto-discover")
    p.add_argument("--baseline", default="cuda12")
    p.add_argument("--native13", default="cuda13-native", help="same-toolkit native condition")
    args = p.parse_args()

    conditions = args.conditions or discover_conditions(args.results_dir)
    if not conditions:
        raise SystemExit(f"no results found under {args.results_dir}")
    if args.baseline not in conditions:
        print(f"baseline '{args.baseline}' not found; using '{conditions[0]}'")
        args.baseline = conditions[0]

    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dgemm = load(args.results_dir, conditions, "dgemm")
    pde = load(args.results_dir, conditions, "pde")

    figs = []
    if dgemm:
        figs += plot_dgemm(dgemm, conditions, args.baseline, fig_dir)
    if pde:
        figs += plot_pde(pde, conditions, args.baseline, fig_dir)
    report = write_report(args, conditions, dgemm, pde, figs)
    print(f"wrote {report} and {len(figs)} figures in {fig_dir}")


if __name__ == "__main__":
    main()
