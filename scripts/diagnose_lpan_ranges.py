#!/usr/bin/env python3
"""Print actual activation ranges (softmax-input, GELU-input, LN-var) seen by
the trained LPAN model on real GLUE inputs, and compare them to the polynomial
approximation intervals exported in `results/coefficients/`.

Out-of-range slots blow up degree-12 polys catastrophically; this script tells
us *which* layer × *which* op is the offender so we know where to refit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--task", default="sst2")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    ckpt = MULTI_MODEL_DIR / args.model / "staged_lpan_final" / "best_model"
    if not ckpt.exists():
        ckpt = MULTI_MODEL_DIR / args.model / "stage4_range_aware" / "best_model"

    print(f"[diag] model={args.model} ({cfg['short']})  task={args.task}  ckpt={ckpt}")

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.models.lpan_loader import load_lpan_model

    model = load_lpan_model(args.model, ckpt, num_labels=2)
    model.eval()
    coeffs = load_coefficients(args.model, task=args.task)

    tok = AutoTokenizer.from_pretrained(cfg["name"])
    if args.task == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        text = lambda r: (r["sentence"], None)
    elif args.task == "mrpc":
        ds = load_dataset("glue", "mrpc", split="validation")
        text = lambda r: (r["sentence1"], r["sentence2"])
    else:
        raise ValueError(args.task)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.num_samples, replace=False).tolist()

    # Hook every poly module: capture *input tensor* min/max across the batch.
    captured: dict[str, list[float]] = {}

    def make_hook(name: str):
        def _hook(_mod, inputs, _out):
            x = inputs[0].detach().float()
            mn, mx = float(x.min().item()), float(x.max().item())
            lst = captured.setdefault(name, [+1e30, -1e30])
            lst[0] = min(lst[0], mn)
            lst[1] = max(lst[1], mx)
        return _hook

    for li, blk in enumerate(model.bert.encoder.layer):
        blk.intermediate.intermediate_act_fn.register_forward_hook(
            make_hook(f"L{li}_GELU_in")
        )
        blk.attention.self.poly_softmax.register_forward_hook(
            make_hook(f"L{li}_Softmax_in")
        )
        # LN: hook on the inv-sqrt poly only
        blk.attention.output.LayerNorm.register_forward_hook(
            make_hook(f"L{li}_AttnLN_in")
        )
        blk.output.LayerNorm.register_forward_hook(
            make_hook(f"L{li}_FFNLN_in")
        )

    print(f"\n[diag] running {args.num_samples} {args.task} samples (seq_len={args.seq_len}) …")
    with torch.no_grad():
        for i, idx in enumerate(indices):
            t1, t2 = text(ds[int(idx)])
            kwargs = dict(
                return_tensors="pt", truncation=True, padding="max_length",
                max_length=args.seq_len,
            )
            enc = tok(t1, t2, **kwargs) if t2 is not None else tok(t1, **kwargs)
            _ = model(**enc).logits

    # Report
    print("\n" + "=" * 78)
    print(f"{'op':<20s} {'observed [min, max]':>26s}    {'fitted [a, b]':>22s}   {'OK?':>4}")
    print("=" * 78)
    for li in range(len(model.bert.encoder.layer)):
        # observed
        for tag in ("GELU", "Softmax", "AttnLN", "FFNLN"):
            key = f"L{li}_{tag}_in"
            if key not in captured:
                continue
            mn, mx = captured[key]
            # fitted interval — for LN the polynomial input is the *variance*,
            # but our hook captures the LayerNorm input, so we just dump it.
            fit = ""
            if tag == "GELU" and "GELU" in coeffs[li]:
                a, b = coeffs[li]["GELU"].interval
                fit = f"[{a:7.3f}, {b:7.3f}]"
            elif tag == "Softmax" and "Softmax" in coeffs[li]:
                a, b = coeffs[li]["Softmax"].interval
                fit = f"[{a:7.3f}, {b:7.3f}]"
            elif tag.endswith("LN") and "LN" in coeffs[li]:
                a, b = coeffs[li]["LN"].interval
                fit = f"[{a:7.3f}, {b:7.3f}]"

            ok = "?"
            if tag in ("GELU", "Softmax") and fit:
                a = coeffs[li][tag].interval[0]
                b = coeffs[li][tag].interval[1]
                ok = "✓" if (mn >= a and mx <= b) else "✗"
            print(
                f"{key:<20s} [{mn:9.3f}, {mx:9.3f}]    {fit:>22s}   {ok:>4}"
            )
    print("=" * 78)
    print("\nLegend: '✗' means observed activations exceed the polynomial's fitted interval —")
    print("        a degree-12 poly evaluated outside [a,b] explodes catastrophically.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
