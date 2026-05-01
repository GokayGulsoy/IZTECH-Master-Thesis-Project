#!/usr/bin/env python3
"""HyPER-LPAN unified training entry point.

Single resumable, YAML-driven pipeline that takes ``bert-base-uncased``
through every stage (LPAN baseline → mid-layer Quad replacement →
early-layer LinearMixing replacement → global fine-tuning) in one
invocation.

Usage
-----
::

    # Run from a YAML config (preferred)
    python experiments/train_hyper_lpan.py --config configs/hyper_lpan/sst2_base.yaml

    # CLI overrides take precedence over YAML
    python experiments/train_hyper_lpan.py --config configs/hyper_lpan/sst2_base.yaml \\
        --epochs-per-layer 6 --gamma 2.0

    # Or pure CLI (uses HyperLPANConfig defaults)
    python experiments/train_hyper_lpan.py --task sst2 --model base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fhe_thesis.pipelines import HyperLPANConfig, HyperLPANPipeline


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HyPER-LPAN unified training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (CLI flags override YAML values)")

    # Task & model
    p.add_argument("--model", type=str)
    p.add_argument("--task", type=str)
    p.add_argument("--max-seq-len", type=int)
    p.add_argument("--seed", type=int)

    # Layer composition
    p.add_argument("--linear-mixing-layers", type=str,
                   help="Comma-separated layer indices, e.g. 0,1,2,3")
    p.add_argument("--quad-attention-layers", type=str,
                   help="Comma-separated layer indices, e.g. 4,5,6,7")
    p.add_argument("--quad-num-heads", type=int)

    # Stages B & C
    p.add_argument("--epochs-per-layer", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--gamma-decay", action=argparse.BooleanOptionalAction)
    p.add_argument("--stage-order", choices=["quad_first", "linear_first"])
    p.add_argument("--lr-schedule",
                   choices=["linear", "cosine", "constant_with_warmup"])

    # Stage D
    p.add_argument("--final-epochs", type=int)
    p.add_argument("--global-gamma", type=float)
    p.add_argument("--global-lr-div", type=float)

    # Stage A
    p.add_argument("--lpan-checkpoint", type=str,
                   help="Override path to LPAN best_model directory")
    p.add_argument("--auto-train-lpan", action=argparse.BooleanOptionalAction)

    # Skip flags
    p.add_argument("--skip-stage-b", action="store_true")
    p.add_argument("--skip-stage-c", action="store_true")
    p.add_argument("--skip-stage-d", action="store_true")

    # Output
    p.add_argument("--output-dir", type=str)

    # Run control
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cpu (auto-detect if omitted)")

    return p


# CLI argparse field name → HyperLPANConfig attribute name (dashes vs underscores)
_LIST_FIELDS = {"linear_mixing_layers", "quad_attention_layers"}


def main() -> None:
    args = build_argparser().parse_args()

    # 1. Load YAML (or start with defaults)
    if args.config is not None:
        cfg = HyperLPANConfig.from_yaml(args.config)
    else:
        cfg = HyperLPANConfig()

    # 2. Apply CLI overrides (only fields the user actually passed)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in {"config", "device"}}
    # argparse 'store_true' flags default to False; we only override if True
    for flag in ("skip_stage_b", "skip_stage_c", "skip_stage_d"):
        if not getattr(args, flag):
            overrides.pop(flag, None)
    for field_name, value in overrides.items():
        if field_name in _LIST_FIELDS and isinstance(value, str):
            value = _parse_int_list(value)
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, value)
    # Re-validate (e.g. layer overlap) after overrides
    cfg.__post_init__()

    # 3. Run
    pipeline = HyperLPANPipeline(cfg, device=args.device)
    pipeline.run()


if __name__ == "__main__":
    main()
