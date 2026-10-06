"""Aggregate results/<condition>/{dgemm,pde}.json into PNG figures and a Markdown report.

Usage:
    python src/analyze_results.py --results-dir results \
        --conditions cuda12 cuda13-native cuda13-emu cuda13-emu-eager cpu-numpy cpu-eigen

CPU conditions contain one record per thread count; the fastest one represents the
condition in figures and main tables, and all of them are listed in the thread-sweep table.
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

PREFERRED_ORDER = ["cuda12", "cuda12.4", "cuda13-native", "cuda13-emu", "cuda13-emu-eager", "cpu-numpy", "cpu-eigen"]
CPU_NUMPY = "cpu-numpy"


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


def best_by(rows: list[tuple[tuple, dict]], time_key: str) -> dict[tuple, dict]:
    """Keep the fastest record per key (CPU conditions have one record per thread count)."""
    out: dict[tuple, dict] = {}
    for key, r in rows:
        if key not in out or r[time_key] < out[key][time_key]:
            out[key] = r
    return out


def own_scaled_err(ex: dict) -> float:
    """Scaled error of the condition's own result ('gpu' in GPU records, 'result' in CPU records)."""
    return (ex.get("result") or ex["gpu"])["max_scaled_err_in_u"]


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


def new_figure(panels: int = 1, max_cols: int = 3):
    """Create a figure with `panels` axes wrapped into rows of at most max_cols."""
    ncols = min(panels, max_cols)
    nrows = -(-panels // ncols)
    fig, grid = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.4 * nrows), facecolor=SURFACE, squeeze=False)
    axes = list(grid.flat)
    for ax in axes[panels:]:
        ax.set_visible(False)
    for ax in axes[:panels]:
        style_axes(ax)
    return fig, axes[:panels]


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
    return best_by([((c, r["phi"], r["n"]), r) for c, p in data.items() for r in ok_rows(p)], "median_s")


def speedup_figure(lut: dict, conditions: list[str], ref: str, panels: list, sizes: list[int], time_key: str,
                   panel_title, xlabel: str, path: Path, title: str) -> bool:
    """Speedup relative to `ref` vs size, one panel per first key (phi or scheme)."""
    others = [c for c in conditions if c != ref]
    if not any(k[0] == ref for k in lut) or not others:
        return False
    fig, axes = new_figure(len(panels))
    for ax, panel in zip(axes, panels):
        for cond in others:
            pts = [
                (n, lut[(ref, panel, n)][time_key] / lut[(cond, panel, n)][time_key])
                for n in sizes
                if (cond, panel, n) in lut and (ref, panel, n) in lut
            ]
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        ax.axhline(1.0, color=REFERENCE_GRAY, linewidth=1, linestyle="--")
        set_size_ticks(ax, sizes)
        speedup_axis(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"speedup vs {ref}")
        ax.set_title(panel_title(panel))
        legend(ax)
    finish(fig, path, title)
    return True


def plot_dgemm(data: dict[str, dict], conditions: list[str], baseline: str, cpu_ref: str,
               fig_dir: Path) -> list[str]:
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

    # Speedup vs the GPU baseline and vs the CPU reference
    for ref, name in ((baseline, "dgemm_speedup.png"), (cpu_ref, "dgemm_speedup_vs_cpu.png")):
        if speedup_figure(lut, conditions, ref, phis, sizes, "median_s", lambda phi: f"phi = {phi:g}",
                          "matrix size N", fig_dir / name, f"DGEMM speedup relative to {ref}"):
            figs.append(name)

    # Accuracy against the correctly rounded reference
    fig, axes = new_figure(len(phis))
    for ax, phi in zip(axes, phis):
        numpy_ref: dict[int, float] = {}  # identical inputs in every condition, so keep the first
        for cond in conditions:
            pts = []
            for n in sizes:
                ex = lut.get((cond, phi, n), {}).get("vs_exact_sampled")
                if ex:
                    pts.append((n, own_scaled_err(ex)))
                    if "numpy" in ex:
                        numpy_ref.setdefault(n, ex["numpy"]["max_scaled_err_in_u"])
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        if numpy_ref and CPU_NUMPY not in data:  # the cpu-numpy series already shows it
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


def plot_dgemm_phi(data: dict[str, dict], conditions: list[str], fig_dir: Path) -> list[str]:
    """Throughput and accuracy as a function of the exponent-range parameter phi."""
    lut = dgemm_lookup(data)
    phis = sorted({k[1] for k in lut})
    sizes = sorted({k[2] for k in lut})
    if len(phis) < 2:
        return []

    def phi_axis(ax) -> None:
        ax.set_xticks(phis)
        ax.set_xticklabels([f"{p:g}" for p in phis])
        ax.set_xlabel("phi (exponent range of the inputs)")

    fig, axes = new_figure(len(sizes))
    for ax, n in zip(axes, sizes):
        for cond in conditions:
            pts = [(phi, lut[(cond, phi, n)]["tflops_median"]) for phi in phis if (cond, phi, n) in lut]
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        phi_axis(ax)
        ax.set_yscale("log")
        ax.set_ylabel("TFLOPS (median)")
        ax.set_title(f"N = {n}")
        legend(ax)
    finish(fig, fig_dir / "dgemm_phi_tflops.png", "DGEMM throughput vs input exponent range")

    fig, axes = new_figure(len(sizes))
    for ax, n in zip(axes, sizes):
        numpy_ref: dict[float, float] = {}
        for cond in conditions:
            pts = []
            for phi in phis:
                ex = lut.get((cond, phi, n), {}).get("vs_exact_sampled")
                if ex:
                    pts.append((phi, own_scaled_err(ex)))
                    if "numpy" in ex:
                        numpy_ref.setdefault(phi, ex["numpy"]["max_scaled_err_in_u"])
            if pts:
                ax.plot(*zip(*pts), **series_style(conditions, cond))
        if numpy_ref and CPU_NUMPY not in data:
            ax.plot(*zip(*sorted(numpy_ref.items())), color=REFERENCE_GRAY, linestyle="--", linewidth=1.5,
                    label="NumPy (CPU)")
        phi_axis(ax)
        ax.set_yscale("log")
        ax.set_ylabel("max scaled error [u = 2^-53]")
        ax.set_title(f"N = {n}")
        legend(ax)
    finish(fig, fig_dir / "dgemm_phi_error.png", "DGEMM accuracy vs input exponent range")
    return ["dgemm_phi_tflops.png", "dgemm_phi_error.png"]


def dgemm_table(data: dict[str, dict], conditions: list[str], baseline: str, native13: str, cpu_ref: str) -> str:
    lut = dgemm_lookup(data)
    lines = [
        f"| phi | N | condition | threads | median [s] | TFLOPS | speedup vs {baseline} | speedup vs {native13} "
        f"| speedup vs {cpu_ref} | rel err vs NumPy | max scaled err [u] (result / NumPy) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(k[1], k[2]) for k in lut})
    for phi, n in keys:
        for cond in conditions:
            r = lut.get((cond, phi, n))
            if r is None:
                failed = [x for x in data.get(cond, {}).get("results", []) if x["n"] == n and x["phi"] == phi]
                if failed:
                    lines.append(f"| {phi:g} | {n} | {cond} | | failed | | | | | | `{failed[0].get('error', '')[:60]}` |")
                continue
            sp_base = _ratio(lut.get((baseline, phi, n)), r)
            sp_nat = _ratio(lut.get((native13, phi, n)), r)
            sp_cpu = _ratio(lut.get((cpu_ref, phi, n)), r)
            rel = r.get("vs_numpy", {}).get("max_rel_err")
            ex = r.get("vs_exact_sampled", {})
            own_u = own_scaled_err(ex) if ex else None
            np_u = ex.get("numpy", {}).get("max_scaled_err_in_u")
            lines.append(
                f"| {phi:g} | {n} | {cond} | {r.get('threads', '-')} | {r['median_s']:.4f} | {r['tflops_median']:.3f} "
                f"| {sp_base} | {sp_nat} | {sp_cpu} | {_sci(rel)} | {_num(own_u)} / {_num(np_u)} |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PDE
# --------------------------------------------------------------------------- #
def pde_lookup(data: dict[str, dict]) -> dict[tuple[str, str, int], dict]:
    return best_by([((c, r["scheme"], r["n"]), r) for c, p in data.items() for r in ok_rows(p)], "time_per_step_s")


def plot_pde(data: dict[str, dict], conditions: list[str], baseline: str, cpu_ref: str, fig_dir: Path) -> list[str]:
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

    for ref, name in ((baseline, "pde_speedup.png"), (cpu_ref, "pde_speedup_vs_cpu.png")):
        if speedup_figure(lut, conditions, ref, schemes, sizes, "time_per_step_s", str, "grid size N (N x N)",
                          fig_dir / name, f"2D heat equation: speedup relative to {ref}"):
            figs.append(name)
    return figs


def pde_table(data: dict[str, dict], conditions: list[str], baseline: str, native13: str, cpu_ref: str) -> str:
    lut = pde_lookup(data)
    lines = [
        f"| scheme | N | steps | condition | threads | ms/step | TFLOPS | speedup vs {baseline} "
        f"| speedup vs {native13} | speedup vs {cpu_ref} | rel err vs discrete exact | rel err vs NumPy "
        "| discretization err |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme, n in sorted({(k[1], k[2]) for k in lut}):
        for cond in conditions:
            r = lut.get((cond, scheme, n))
            if r is None:
                continue
            sp = [
                f"{ref['time_per_step_s'] / r['time_per_step_s']:.2f}x" if ref else "-"
                for ref in (lut.get((c, scheme, n)) for c in (baseline, native13, cpu_ref))
            ]
            lines.append(
                f"| {scheme} | {n} | {r['steps']} | {cond} | {r.get('threads', '-')} | {r['time_per_step_s'] * 1e3:.3f} "
                f"| {r['tflops']:.4f} | {' | '.join(sp)} | {_sci(r['vs_discrete_exact']['max_rel_err'])} "
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
        if cond not in payloads or "gpu_name" not in payloads[cond]["env"]:
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


def cpu_env_table(payloads: dict[str, dict], conditions: list[str]) -> str:
    rows = [c for c in conditions if c in payloads and "cpu_model" in payloads[c]["env"]]
    if not rows:
        return ""
    lines = [
        "| condition | CPU | cores | Eigen (AVX2 / FMA) | compiler | BLAS (NumPy) | OpenMP binding |",
        "|---|---|---:|---|---|---|---|",
    ]
    for cond in rows:
        e = payloads[cond]["env"]
        cpp = e.get("cpp", {})
        blas = ", ".join(f"{p['internal_api']} {p['version']} ({p.get('architecture')})" for p in e.get("threadpools", []))
        omp = ", ".join(f"{k}={v}" for k, v in e.get("omp_env", {}).items() if v)
        lines.append(
            f"| {cond} | {e['cpu_model']} | {e['cpu_count']} | {cpp.get('eigen_version', '-')} "
            f"({cpp.get('avx2')} / {cpp.get('eigen_fma')}) | g++ {cpp.get('compiler', '-')} | {blas or '-'} | {omp or '-'} |"
        )
    return "\n".join(lines)


def thread_sweep_table(dgemm: dict[str, dict], pde: dict[str, dict]) -> str:
    """All CPU records (one per thread count), the fastest per case marked with *."""
    lines = ["| benchmark | case | condition | threads | median [s] | TFLOPS |", "|---|---|---|---:|---:|---:|"]
    found = False
    for bench, data, key, time_key, tf_key in (
        ("dgemm", dgemm, lambda r: f"phi={r['phi']:g}, N={r['n']}", "median_s", "tflops_median"),
        ("pde", pde, lambda r: f"{r['scheme']}, N={r['n']}", "time_per_step_s", "tflops"),
    ):
        for cond, payload in data.items():
            rows = [r for r in ok_rows(payload) if "threads" in r]
            best = best_by([((key(r),), r) for r in rows], time_key)
            for r in sorted(rows, key=lambda r: (key(r), r["threads"])):
                found = True
                mark = " *" if best[(key(r),)] is r else ""
                lines.append(
                    f"| {bench} | {key(r)} | {cond} | {r['threads']}{mark} | {r['median_s']:.4f} | {r[tf_key]:.4f} |"
                )
    return "\n".join(lines) if found else ""


def write_report(args, conditions, dgemm, pde, figs) -> Path:
    any_payload = {**pde, **dgemm}
    parts = [
        "# CUDA 12 vs CUDA 13.4 (Ozaki scheme) FP64 benchmark report",
        "",
        f"- conditions: {', '.join(conditions)}",
        f"- baseline for speedup: `{args.baseline}`; emulation effect isolated against `{args.native13}`; "
        f"CPU reference `{args.cpu_ref}`",
        "",
        "## Environment",
        "",
        env_table(any_payload, conditions),
        "",
        cpu_env_table(any_payload, conditions),
        "",
        "Speedups are ratios of median times. Errors: *rel err vs NumPy* = max|C-C_np| / max|C_np|;",
        "*scaled err* = max over sampled entries of |c_ij - exact_ij| / (|A||B|)_ij in units of u = 2^-53.",
        "",
    ]
    if dgemm:
        parts += ["## DGEMM", ""]
        parts += [f"![{f}](figures/{f})" for f in figs if f.startswith("dgemm")]
        parts += ["", dgemm_table(dgemm, conditions, args.baseline, args.native13, args.cpu_ref), ""]
    if pde:
        parts += ["## 2D heat equation", ""]
        parts += [f"![{f}](figures/{f})" for f in figs if f.startswith("pde")]
        parts += ["", pde_table(pde, conditions, args.baseline, args.native13, args.cpu_ref), ""]
    sweep = thread_sweep_table(dgemm, pde)
    if sweep:
        parts += ["## CPU thread sweep", "", "`*` = fastest configuration, used in the figures and tables above.",
                  "", sweep, ""]
    out = args.results_dir / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--conditions", nargs="+", default=None, help="order defines colors; default: auto-discover")
    p.add_argument("--baseline", default="cuda12")
    p.add_argument("--native13", default="cuda13-native", help="same-toolkit native condition")
    p.add_argument("--cpu-ref", default="cpu-eigen", help="CPU condition for the speedup-vs-CPU figures")
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
        figs += plot_dgemm(dgemm, conditions, args.baseline, args.cpu_ref, fig_dir)
        figs += plot_dgemm_phi(dgemm, conditions, fig_dir)
    if pde:
        figs += plot_pde(pde, conditions, args.baseline, args.cpu_ref, fig_dir)
    report = write_report(args, conditions, dgemm, pde, figs)
    print(f"wrote {report} and {len(figs)} figures in {fig_dir}")


if __name__ == "__main__":
    main()
