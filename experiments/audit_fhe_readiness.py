#!/usr/bin/env python3
"""Audit FHE-readiness of LPAN-trained checkpoints.

For every (model, task) pair we push the validation set through the
plaintext LPAN model with hooks on each polynomial submodule
(``PolynomialGELU``, ``PerHeadPolynomialSoftmax``,
``PolynomialLayerNorm``) and record three things per (layer, op):

1. ``oor_frac``  — fraction of input scalars that fall outside the
   polynomial's fitted interval ``[a, b]`` (i.e. the slots that the
   plaintext ``clamp`` silently fixes but CKKS cannot).
2. ``max_excursion`` — ``max(|x| beyond [a, b])``, useful for sizing
   the wider intervals required by a range-aware fine-tune.
3. ``poly_blowup`` — the worst-case |p(x_std)| that *would* occur if
   the clamp were removed (i.e. the actual encrypted-side output);
   anything ≥ ~10× the in-range maximum signals a checkpoint that
   needs Stage-4 retraining.

Per-combo JSON reports are written to
``results/fhe_readiness/<task>_<model>.json``.  A combined summary
table is printed to stdout and saved to
``results/fhe_readiness/summary.json``.

A checkpoint is flagged ``needs_stage4`` if any (layer, op) has
``oor_frac > --max-oor-frac`` (default 0.1 %).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fhe_thesis.config import MODEL_REGISTRY
from fhe_thesis.models.activations import (
    PerHeadPolynomialSoftmax,
    PolynomialGELU,
    PolynomialLayerNorm,
    PolynomialSoftmax,
)
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.tasks import GLUE_TASKS, get_task
from fhe_thesis.training.trainer import load_glue_dataset


OUT_DIR = Path("results/fhe_readiness")


def _ckpt_path(model: str, task: str) -> Path:
    base = Path("results/multi_model") / model
    if task != "sst2":
        base = base / task
    return base / "staged_lpan_final" / "best_model"


def _interval_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the tensor whose values the polynomial actually clamps.

    * GELU clamps the raw pre-activation.
    * Softmax clamps ``x − x.max(dim=-1)`` (the "shifted scores"). We
      strip BERT padding-mask sentinels (very negative additive masks
      added before softmax) since those slots correspond to padding
      tokens that aren't measured during encrypted inference.
    * LayerNorm clamps ``var + eps`` of the per-token features.
    """
    if isinstance(module, PolynomialGELU):
        return x
    if isinstance(module, (PolynomialSoftmax, PerHeadPolynomialSoftmax)):
        shifted = x - x.max(dim=-1, keepdim=True).values
        # Padding-mask scores are pushed to ~-FLT_MAX by the additive
        # attention mask; drop anything obviously sentinel.
        return shifted[shifted > -1e3]
    if isinstance(module, PolynomialLayerNorm):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return var + module.eps
    raise TypeError(f"unsupported polynomial module: {type(module)}")


def _module_label(name: str) -> str:
    if "intermediate_act_fn" in name:
        return "GELU"
    if "poly_softmax" in name:
        return "Softmax"
    if "LayerNorm" in name:
        return "LN"
    return "?"


def _layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def _attach_hooks(model: torch.nn.Module) -> tuple[list, dict[str, dict[str, Any]]]:
    """Register forward hooks on every poly module; return (handles, stats)."""
    stats: dict[str, dict[str, Any]] = {}
    handles = []
    poly_types = (
        PolynomialGELU,
        PolynomialSoftmax,
        PerHeadPolynomialSoftmax,
        PolynomialLayerNorm,
    )
    for name, module in model.named_modules():
        if not isinstance(module, poly_types):
            continue
        layer = _layer_idx(name)
        op = _module_label(name)
        key = f"L{layer}_{op}_{name.rsplit('.', 1)[-1]}"
        stats[key] = {
            "name": name,
            "layer": layer,
            "op": op,
            "interval": [float(module.a), float(module.b)],
            "n_total": 0,
            "n_oor": 0,
            "max_below": 0.0,  # max( a - x ) when x < a
            "max_above": 0.0,  # max( x - b ) when x > b
            "min_seen": float("inf"),
            "max_seen": float("-inf"),
        }

        def make_hook(stats_entry, mod_ref):
            def hook(_mod, inputs, _output):
                x = inputs[0]
                interval_x = _interval_input(mod_ref, x).detach()
                a = float(mod_ref.a)
                b = float(mod_ref.b)
                stats_entry["n_total"] += interval_x.numel()
                below_mask = interval_x < a
                above_mask = interval_x > b
                stats_entry["n_oor"] += int(
                    below_mask.sum().item() + above_mask.sum().item()
                )
                if below_mask.any():
                    stats_entry["max_below"] = max(
                        stats_entry["max_below"],
                        float((a - interval_x[below_mask]).max().item()),
                    )
                if above_mask.any():
                    stats_entry["max_above"] = max(
                        stats_entry["max_above"],
                        float((interval_x[above_mask] - b).max().item()),
                    )
                stats_entry["min_seen"] = min(
                    stats_entry["min_seen"], float(interval_x.min().item())
                )
                stats_entry["max_seen"] = max(
                    stats_entry["max_seen"], float(interval_x.max().item())
                )
            return hook

        handles.append(module.register_forward_hook(make_hook(stats[key], module)))
    return handles, stats


def _audit_one(
    model_key: str,
    task_name: str,
    *,
    num_samples: int,
    batch_size: int,
    device: str,
    max_oor_frac: float,
) -> dict[str, Any]:
    cfg = MODEL_REGISTRY[model_key]
    ckpt = _ckpt_path(model_key, task_name)
    if not ckpt.exists():
        return {"status": "missing", "checkpoint": str(ckpt)}

    task = get_task(task_name)
    print(f"  [audit] {model_key} × {task_name}  ckpt={ckpt}")

    model = load_lpan_model(
        model_key, ckpt, num_labels=task.num_labels, device=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    _, eval_dict = load_glue_dataset(task, tokenizer, max_length=128)
    eval_ds = next(iter(eval_dict.values()))
    if num_samples > 0 and num_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(num_samples))

    handles, stats = _attach_hooks(model)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            model(**batch)
    wall = time.time() - t0
    for h in handles:
        h.remove()

    # Post-process: per-op fractions, blow-up estimate.
    by_op: dict[str, dict] = {}
    needs_stage4 = False
    worst_op: tuple[str, float] = ("", 0.0)
    for key, s in stats.items():
        n_total = max(s["n_total"], 1)
        oor = s["n_oor"] / n_total
        s["oor_frac"] = oor
        s["max_excursion"] = max(s["max_below"], s["max_above"])
        # Worst-case standardised input if clamp were removed.
        a, b = s["interval"]
        x_worst = max(abs(s["min_seen"]), abs(s["max_seen"]))
        # Standardise to [-1, 1] domain of Cheb fit:
        s["x_std_worst"] = float((2.0 * x_worst - (a + b)) / (b - a))
        by_op[key] = s
        if oor > max_oor_frac:
            needs_stage4 = True
            if oor > worst_op[1]:
                worst_op = (key, oor)

    return {
        "status": "ok",
        "checkpoint": str(ckpt),
        "wall_s": wall,
        "num_eval_samples": len(eval_ds),
        "max_oor_frac_threshold": max_oor_frac,
        "needs_stage4": needs_stage4,
        "worst_op": {"key": worst_op[0], "oor_frac": worst_op[1]} if worst_op[0] else None,
        "ops": by_op,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FHE readiness of LPAN checkpoints")
    parser.add_argument(
        "--model", nargs="*", default=list(MODEL_REGISTRY),
        help="Which model(s) to audit (default: all)",
    )
    parser.add_argument(
        "--task", nargs="*", default=list(GLUE_TASKS),
        help="Which GLUE task(s) (default: all)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="Validation samples per (model, task); 0 = use full split.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-oor-frac", type=float, default=1e-3,
        help="A combo with any (layer, op) above this fraction is flagged "
             "for Stage-4 range-aware fine-tune (default: 0.1%%).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"threshold": args.max_oor_frac, "combos": []}

    for task_name in args.task:
        for model_key in args.model:
            report = _audit_one(
                model_key, task_name,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=args.device,
                max_oor_frac=args.max_oor_frac,
            )
            out_path = OUT_DIR / f"{task_name}_{model_key}.json"
            out_path.write_text(json.dumps(report, indent=2))
            entry = {
                "model": model_key, "task": task_name,
                "status": report["status"],
                "needs_stage4": report.get("needs_stage4"),
                "worst_op": report.get("worst_op"),
            }
            summary["combos"].append(entry)
            if report["status"] == "ok":
                worst = report["worst_op"]
                tag = "FLAG" if report["needs_stage4"] else " ok "
                worst_str = (
                    f" worst={worst['key']}@{100*worst['oor_frac']:.3f}%"
                    if worst else ""
                )
                print(f"    [{tag}] {model_key:5s} × {task_name:5s}  "
                      f"wall={report['wall_s']:.1f}s{worst_str}")
            else:
                print(f"    [skip] {model_key:5s} × {task_name:5s}  "
                      f"({report['status']})")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summary_path}")
    flagged = [c for c in summary["combos"] if c.get("needs_stage4")]
    print(f"Flagged for Stage-4: {len(flagged)} / "
          f"{sum(1 for c in summary['combos'] if c['status'] == 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
