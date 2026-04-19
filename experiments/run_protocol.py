#!/usr/bin/env python3
"""Unified LPAN-FHE protocol runner — replaces experiments/12_*, 13_*.

Runs an encrypted forward pass for any BERT variant in
``MODEL_REGISTRY`` and dumps a JSON report to
``results/encrypted_inference/<model>_<phase>.json``.

Usage
-----
    python experiments/run_protocol.py \
        --model {tiny|mini|small|base} \
        --phase {ffn|attention|layer|model} \
        [--seq-len 8] [--layer 0] \
        [--checkpoint <path>]

Phases
------
* ``ffn``       — single FFN+LN block (Phase 1 smoke-test)
* ``attention`` — single MHA+LN block (Phase 2 smoke-test)
* ``layer``     — full transformer layer (Phase 3)
* ``model``     — full encoder + classifier head (Phase 3+)

Notes
-----
* Heavy-compute machine only (TenSEAL N=16384 / 32768).
* Coefficients are pulled from ``results/coefficients/<model>.json``
  if present (extracted by ``extract_coefficients.py``); otherwise
  fall back to profile-and-fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhe_thesis.config import (  # noqa: E402
    ENCRYPTED_INFERENCE_DIR,
    MODEL_REGISTRY,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument(
        "--phase",
        required=True,
        choices=["ffn", "attention", "layer", "model"],
    )
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument(
        "--layer",
        type=int,
        default=0,
        help="layer index for ffn/attention/layer phases",
    )
    p.add_argument(
        "--checkpoint", default=None, help="optional LPAN-trained checkpoint path"
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    hidden = cfg["hidden"]
    print(
        f"[run_protocol] model={args.model} ({cfg['short']}, "
        f"hidden={hidden}, heads={cfg['heads']}, layers={cfg['layers']})"
    )
    print(f"[run_protocol] phase={args.phase}  seq_len={args.seq_len}")

    # Lazy import: tenseal/transformers are heavy and only needed here.
    from fhe_thesis.encryption import TenSEALBackend
    from fhe_thesis.encryption.context import make_context
    from fhe_thesis.encryption.depth import (
        DepthAudit,
        transformer_layer_depth,
    )
    from fhe_thesis.encryption.protocol import run_phase

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.seq_len, hidden)).astype(np.float64) * 0.1

    print("[1/3] Booting CKKS context (this can take a while on first call)…")
    ctx = make_context()
    backend = TenSEALBackend(ctx)
    print(f"      capabilities: {backend.capabilities}")

    audit = DepthAudit(initial_levels=backend.capabilities.initial_levels)
    print(
        f"\n[2/3] Static depth budget: full layer = "
        f"{transformer_layer_depth()} levels "
        f"(have {audit.initial_levels})"
    )

    print("\n[3/3] Encrypted run …")
    out, timings = run_phase(
        args.phase,
        args.model,
        backend,
        x,
        layer_idx=args.layer,
        checkpoint_path=args.checkpoint,
    )
    print(f"      output shape: {out.shape}")
    print(f"      total wall: {timings['total']:.2f}s")

    report = {
        "model": args.model,
        "phase": args.phase,
        "seq_len": args.seq_len,
        "hidden": hidden,
        "num_heads": cfg["heads"],
        "checkpoint": args.checkpoint,
        "static_layer_depth": transformer_layer_depth(),
        "backend_initial_levels": backend.capabilities.initial_levels,
        "timings_sec": timings,
        "output_norm": float(np.linalg.norm(out)),
    }
    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.phase}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n→ wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
