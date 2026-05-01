"""Ext 3 driver — run task-adaptive composition selection on a GLUE dev split.

Two selectors are available:

* ``--method entropy`` (default): cheap, task-agnostic two-threshold
  rule using normalised attention entropy as a proxy.
* ``--method mckp``: provably optimal Multiple-Choice Knapsack DP over
  measured per-layer attention substitution errors. Slower but exact
  with respect to its surrogate objective.

Usage::

    python experiments/select_composition.py \\
        --model base --task mrpc \\
        --checkpoint <path-to-finetuned-bert> \\
        --n-samples 64 --max-seq-len 64 \\
        --method mckp --budget-fraction 0.5

Writes a JSON plan and prints per-layer details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="HyPER-LPAN composition selector")
    p.add_argument("--model", default="base",
                   choices=["tiny", "mini", "small", "base"])
    p.add_argument("--task", default="sst2",
                   choices=["sst2", "mrpc", "qnli", "rte"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Plaintext fine-tuned checkpoint (default: HF pretrained)")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--method", default="entropy", choices=["entropy", "mckp"],
                   help="Selector: 'entropy' (heuristic) or 'mckp' (provably optimal DP)")
    p.add_argument("--budget-fraction", type=float, default=0.5,
                   help="Budget as fraction of full-LPAN cost (default: 0.5)")
    p.add_argument("--min-lpan", type=int, default=2,
                   help="Minimum number of LPAN layers to keep (default: 2)")
    p.add_argument("--gamma-q", type=float, default=0.5,
                   help="(MCKP) fraction of LM residual that QuadAttention removes")
    p.add_argument("--cost-scale", type=int, default=10,
                   help="(MCKP) cost discretisation factor")
    p.add_argument("--tau-low", type=float, default=None)
    p.add_argument("--tau-high", type=float, default=None)
    p.add_argument("--out", type=str, default="results/composition")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _load_dev(task: str, n_samples: int, max_seq_len: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    text_col = {"sst2": "sentence", "mrpc": "sentence1",
                "qnli": "question", "rte": "sentence1"}[task]
    ds = load_dataset("glue", task, split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    samples = []
    for item in ds:
        enc = tok(item[text_col],
                  max_length=max_seq_len,
                  padding="max_length",
                  truncation=True,
                  return_tensors="pt")
        samples.append({
            "input_ids": enc["input_ids"][0].numpy(),
            "attention_mask": enc["attention_mask"][0].numpy(),
        })
    return samples


def main():
    from transformers import AutoModelForSequenceClassification

    from fhe_thesis.config import MODEL_REGISTRY
    from fhe_thesis.optimization.composition_selector import (
        compose_for_task, compose_for_task_mckp, LAYER_COST,
    )

    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== HyPER-LPAN Composition Selector ({args.method}) ===")
    print(f"  model={args.model} task={args.task} n={args.n_samples} max_len={args.max_seq_len}")

    samples = _load_dev(args.task, args.n_samples, args.max_seq_len)

    cfg = MODEL_REGISTRY[args.model]
    src = args.checkpoint or cfg["name"]
    print(f"  checkpoint: {src}")
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)

    # Determine number of layers from config
    n_layers = model.config.num_hidden_layers
    full_lpan_cost = n_layers * LAYER_COST["L"]
    budget = args.budget_fraction * full_lpan_cost

    if args.method == "entropy":
        ent_budget = (budget if args.tau_low is None and args.tau_high is None
                      else None)
        if ent_budget is not None:
            print(f"  budget={args.budget_fraction:.2f} * full_lpan_cost "
                  f"({full_lpan_cost:.1f}) = {budget:.2f}")
        else:
            print(f"  manual thresholds: tau_low={args.tau_low} tau_high={args.tau_high}")
        plan = compose_for_task(
            model, samples,
            budget=ent_budget,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            min_lpan=args.min_lpan,
            device=args.device,
        )
        score_label = "entropy"
    else:  # mckp
        print(f"  budget={args.budget_fraction:.2f} * full_lpan_cost "
              f"({full_lpan_cost:.1f}) = {budget:.2f}  gamma_q={args.gamma_q}")
        plan = compose_for_task_mckp(
            model, samples,
            budget=budget,
            gamma_q=args.gamma_q,
            min_lpan=args.min_lpan,
            cost_scale=args.cost_scale,
            device=args.device,
        )
        score_label = "LM-drift"

    print(f"\n  per-layer {score_label} scores:")
    for li, h in enumerate(plan.layer_entropies):
        kind = ("LM" if li in plan.linear_mixing_layers
                else "Q" if li in plan.quad_attention_layers
                else "L")
        print(f"    L{li:02d}  s={h:.4f}  -> {kind}")

    print(f"\n  selected plan:")
    print(f"    linear_mixing_layers: {plan.linear_mixing_layers}")
    print(f"    quad_attention_layers: {plan.quad_attention_layers}")
    print(f"    lpan_layers: {plan.lpan_layers}")
    print(f"    estimated_cost: {plan.estimated_cost:.2f} "
          f"(vs full-LPAN {full_lpan_cost:.1f}, "
          f"speedup {full_lpan_cost / max(plan.estimated_cost, 1e-9):.2f}x)")

    out_path = out_dir / f"plan_{args.method}_{args.model}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({
            **plan.to_dict(),
            "method": args.method,
            "model": args.model,
            "task": args.task,
            "n_samples": args.n_samples,
            "budget_fraction": args.budget_fraction,
            "checkpoint": str(src),
        }, f, indent=2)
    print(f"\n  saved -> {out_path}")
    print("\n  yaml fragment:\n")
    print(plan.to_yaml_fragment())


if __name__ == "__main__":
    main()
