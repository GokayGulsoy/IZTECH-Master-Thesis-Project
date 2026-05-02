"""Pareto plot — accuracy vs FHE-cost per task.

Reads results/multi_model/{task}/{backbone}/hyper_lpan_b{NN}_seed{S}/results.json
plus the matching MCKP plan results/composition/plan_mckp_{backbone}_{task}_b{NN}.json
to obtain the estimated cost units, and emits one curve per (backbone, task).

Usage:
    python experiments/plot_pareto.py
    python experiments/plot_pareto.py --backbones base --tasks mrpc sst2
    python experiments/plot_pareto.py --out figures/pareto.pdf
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args():
    p = argparse.ArgumentParser(description="Per-task Pareto curves.")
    p.add_argument("--backbones", nargs="*", default=None,
                   help="Filter backbones (default: all found)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Filter tasks (default: all found)")
    p.add_argument("--results-root", default="results/multi_model")
    p.add_argument("--plans-root", default="results/composition")
    p.add_argument("--out", default="figures/pareto.pdf")
    p.add_argument("--csv", default="figures/pareto.csv")
    p.add_argument("--show-baseline", action="store_true",
                   help="Add LPAN-baseline horizontal line per task")
    return p.parse_args()


def _scan(root: Path, plans: Path,
          backbones: List[str] | None,
          tasks: List[str] | None
          ) -> Dict[Tuple[str, str], List[Tuple[float, float, str]]]:
    """Return {(backbone, task): [(cost, final_acc, btag), ...]} aggregated by btag."""
    pat = re.compile(r"hyper_lpan_(b\d+)_seed(\d+)$")
    raw: Dict[Tuple[str, str, str], Tuple[float | None, List[float]]] = {}
    for results_json in root.glob("*/*/hyper_lpan_b*_seed*/results.json"):
        d = results_json.parent
        m = pat.match(d.name)
        if not m:
            continue
        btag = m.group(1)
        backbone = d.parent.name
        task = d.parent.parent.name
        if backbones and backbone not in backbones:
            continue
        if tasks and task not in tasks:
            continue
        try:
            r = json.loads(results_json.read_text())
        except Exception:
            continue
        final = r.get("final_accuracy")
        if final is None:
            continue
        plan_file = plans / f"plan_mckp_{backbone}_{task}_{btag}.json"
        cost = None
        if plan_file.exists():
            try:
                cost = float(json.loads(plan_file.read_text()).get("estimated_cost"))
            except Exception:
                cost = None
        key = (backbone, task, btag)
        prev = raw.get(key, (cost, []))
        raw[key] = (cost if cost is not None else prev[0], prev[1] + [float(final)])

    grouped: Dict[Tuple[str, str], List[Tuple[float, float, float, float, int, str]]] = defaultdict(list)
    for (bb, task, btag), (cost, vals) in raw.items():
        if cost is None or not vals:
            continue
        med = statistics.median(vals)
        lo = min(vals)
        hi = max(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        grouped[(bb, task)].append((cost, med, lo, hi, len(vals), btag))
    for k in grouped:
        grouped[k].sort()  # by cost asc
    return grouped


def _baseline_acc(root: Path, backbone: str, task: str) -> float | None:
    """Read LPAN baseline accuracy from any matching results.json."""
    for rj in root.glob(f"{task}/{backbone}/hyper_lpan_*/results.json"):
        try:
            r = json.loads(rj.read_text())
            v = r.get("lpan_baseline")
            if v is not None:
                return float(v)
        except Exception:
            pass
    return None


def main() -> None:
    args = _parse_args()
    root = Path(args.results_root)
    plans = Path(args.plans_root)
    grouped = _scan(root, plans, args.backbones, args.tasks)
    if not grouped:
        print("No matching results found. Looked under:", root)
        return

    # CSV dump (always)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("backbone,task,budget_tag,cost_units,n_seeds,acc_median,acc_min,acc_max\n")
        for (bb, task), pts in sorted(grouped.items()):
            for cost, med, lo, hi, n, btag in pts:
                f.write(f"{bb},{task},{btag},{cost:.4f},{n},{med:.6f},{lo:.6f},{hi:.6f}\n")
    print(f"  wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV written, skipping figure.")
        return

    tasks = sorted({t for _, t in grouped.keys()})
    backbones = sorted({b for b, _ in grouped.keys()})
    n = len(tasks)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        for bb in backbones:
            pts = grouped.get((bb, task), [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            yerr_lo = [p[1] - p[2] for p in pts]
            yerr_hi = [p[3] - p[1] for p in pts]
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi],
                        marker="o", capsize=3, label=bb)
        if args.show_baseline:
            for bb in backbones:
                base = _baseline_acc(root, bb, task)
                if base is not None:
                    ax.axhline(base, linestyle="--", alpha=0.4, label=f"{bb} LPAN")
        ax.set_title(task.upper())
        ax.set_xlabel("FHE cost units (estimated)")
        ax.set_ylabel("Final accuracy / F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("HyPER-LPAN per-task Pareto: accuracy vs FHE-cost")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
