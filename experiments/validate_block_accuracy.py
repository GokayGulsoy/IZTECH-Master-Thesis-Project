#!/usr/bin/env python3
"""Per-block PF-SR vs plaintext numerical-accuracy validator.

End-to-end LPAN forward needs ≥31 CKKS multiplicative levels per
transformer layer, which exceeds what TenSEAL's largest supported
ring (``N = 32768``) can deliver without bootstrapping. Until a GPU
CKKS backend with bootstrapping is wired in (see thesis Future Work),
we validate the PF-SR protocol's correctness *per encrypted block*:

* feed the same activation tensor into both the plaintext LPAN module
  and the encrypted block, then
* compare the decrypted output against the plaintext reference under
  L¹ / L² / L∞ metrics and a relative-error percentage.

Usage
-----
    python experiments/validate_block_accuracy.py \
        --model {tiny|mini|small|base} \
        --block {ffn|attention} \
        [--num-samples 10] [--seq-len 8] [--layer 0]

Writes ``results/encrypted_inference/<model>_<block>_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhe_thesis.config import (  # noqa: E402
    ENCRYPTED_INFERENCE_DIR,
    MODEL_REGISTRY,
    MULTI_MODEL_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--block", required=True, choices=["ffn", "attention"])
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="LPAN checkpoint dir; default uses the SST-2 trained model.",
    )
    return p.parse_args()


def _default_checkpoint(model_key: str) -> Path:
    return MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"


def _plaintext_block(model, block: str, layer_idx: int, x_np: np.ndarray) -> np.ndarray:
    """Run the plaintext LPAN block on numpy activations."""
    import torch

    bert_layer = model.bert.encoder.layer[layer_idx]
    x = torch.from_numpy(x_np).float().unsqueeze(0)  # (1, seq, hidden)

    with torch.no_grad():
        if block == "ffn":
            inter = bert_layer.intermediate(x)
            out = bert_layer.output.dense(inter)
            out = bert_layer.output.LayerNorm(out + x)
        elif block == "attention":
            # BERT attention returns (out, attn) — the LayerNorm is in
            # attention.output, applied with residual = x already inside.
            attn_out = bert_layer.attention(x)[0]
            out = attn_out
        else:
            raise ValueError(block)
    return out.squeeze(0).numpy()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    ckpt = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args.model)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"  → run: python run_staged_lpan.py --model {args.model} --task sst2"
        )

    print(f"[validate] model = {args.model} ({cfg['short']})  block = {args.block}")
    print(f"[validate] ckpt  = {ckpt}")
    print(f"[validate] N     = {args.num_samples} samples, seq_len = {args.seq_len}")

    # Heavy imports
    from fhe_thesis.encryption import TenSEALBackend
    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.encryption.depth import DEPTH_COST
    from fhe_thesis.encryption.packing import TokenPackedTensor
    from fhe_thesis.encryption.protocol import (
        encrypt_attention_block,
        encrypt_ffn_block,
        load_model_weights,
    )
    from fhe_thesis.models.lpan_loader import load_lpan_model

    # Compute depth + boot context.
    if args.block == "ffn":
        mult_depth = (
            DEPTH_COST["linear"]
            + DEPTH_COST["polyval_deg6"]
            + DEPTH_COST["linear"]
            + DEPTH_COST["ln_poly"]
        )
    else:
        mult_depth = (
            DEPTH_COST["linear"]
            + DEPTH_COST["qk_scores"]
            + DEPTH_COST["softmax_poly"]
            + DEPTH_COST["attn_apply"]
            + DEPTH_COST["head_concat"]
            + DEPTH_COST["linear"]
            + DEPTH_COST["ln_poly"]
        )

    coeff_mod = [60] + [40] * mult_depth + [60]
    total_bits = sum(coeff_mod)
    poly_mod = 8192 if total_bits <= 218 else (16384 if total_bits <= 438 else 32768)

    print(f"[boot] CKKS context N={poly_mod}, mult_depth={mult_depth}")
    backend = TenSEALBackend(
        poly_modulus_degree=poly_mod, coeff_mod_bit_sizes=coeff_mod
    )

    # Load model + coefficients.
    print("[load] plaintext LPAN model + extracted coefficients …")
    model = load_lpan_model(args.model, ckpt, num_labels=2)
    weights = load_model_weights(args.model, checkpoint_path=str(ckpt))
    coeffs = load_coefficients(args.model)
    layer_w = weights.layers[args.layer]
    layer_c = coeffs[args.layer]

    # ── Sync polynomial intervals from the trained model ───────────
    # The hardcoded `PROFILED_INTERVALS` table only covers a snapshot
    # of BERT-Tiny's first two layers; everything else falls back to
    # L0_GELU/Softmax/LN which can differ wildly from the actual
    # training-time intervals. The trained polynomial submodules carry
    # the correct intervals as `.a, .b` attributes — copy them onto
    # the loaded `PolyCoeffs` so the encrypted side standardises with
    # the same range used at training time.
    bert_layer = model.bert.encoder.layer[args.layer]
    from dataclasses import replace as _dc_replace
    gelu_mod = bert_layer.intermediate.intermediate_act_fn
    layer_c["GELU"] = _dc_replace(
        layer_c["GELU"], interval=(float(gelu_mod.a), float(gelu_mod.b))
    )
    ln_mod = bert_layer.output.LayerNorm
    layer_c["LN"] = _dc_replace(
        layer_c["LN"], interval=(float(ln_mod.a), float(ln_mod.b))
    )
    if "Softmax" in layer_c:
        sm_mod = bert_layer.attention.self.poly_softmax
        layer_c["Softmax"] = _dc_replace(
            layer_c["Softmax"], interval=(float(sm_mod.a), float(sm_mod.b))
        )
    print(
        f"  intervals: GELU={layer_c['GELU'].interval}  "
        f"LN={layer_c['LN'].interval}"
        + (f"  Softmax={layer_c['Softmax'].interval}"
           if 'Softmax' in layer_c else "")
    )

    # Generate samples + run both pipelines.
    # Use real BERT embeddings from SST-2 so activation magnitudes match
    # the training distribution; random Gaussian noise sits outside the
    # LPAN polynomials' fitted intervals and produces meaningless errors.
    print("[data] sampling real SST-2 inputs through BERT embeddings …")
    import torch as _torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["name"])
    ds = load_dataset("glue", "sst2", split="validation")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.num_samples, replace=False).tolist()

    # If validating layer L > 0, we need to push the embeddings through
    # all preceding LPAN encoder layers to get realistic activations.
    samples_x: list[np.ndarray] = []
    with _torch.no_grad():
        for idx in indices:
            enc = tok(
                ds[int(idx)]["sentence"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.seq_len,
            )
            embeds = model.bert.embeddings(enc["input_ids"])
            h = embeds
            for li in range(args.layer):
                h = model.bert.encoder.layer[li](h)[0]
            samples_x.append(h.detach()[0].numpy().astype(np.float64))

    per_sample: list[dict] = []
    enc_wall_total = 0.0

    for i, x in enumerate(samples_x):
        # Plaintext reference.
        y_plain = _plaintext_block(model, args.block, args.layer, x)

        # Encrypted forward.
        ct_x = TokenPackedTensor.encrypt(backend, x)
        t0 = time.time()
        if args.block == "ffn":
            ct_y, _ = encrypt_ffn_block(backend, ct_x, layer_w, layer_c)
        else:
            ct_y, _ = encrypt_attention_block(
                backend, ct_x, layer_w, layer_c, weights.num_heads
            )
        wall = time.time() - t0
        y_dec = ct_y.decrypt(backend)
        enc_wall_total += wall

        diff = y_plain - y_dec
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        linf = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(y_plain), 1e-12))

        per_sample.append(
            {
                "index": i,
                "wall_s": wall,
                "L1": l1,
                "L2": l2,
                "Linf": linf,
                "relative_L2": rel,
            }
        )
        print(
            f"  [{i + 1:>3d}/{args.num_samples}] "
            f"L1={l1:.4f}  L2={l2:.4f}  Linf={linf:.4f}  "
            f"relL2={100 * rel:.2f}%  ({wall:.1f}s)"
        )

    summary = {
        "model": args.model,
        "model_short": cfg["short"],
        "block": args.block,
        "layer": args.layer,
        "seq_len": args.seq_len,
        "num_samples": args.num_samples,
        "checkpoint": str(ckpt),
        "ckks": {
            "poly_modulus_degree": poly_mod,
            "mult_depth": mult_depth,
        },
        "metrics": {
            "mean_L1": float(np.mean([s["L1"] for s in per_sample])),
            "mean_L2": float(np.mean([s["L2"] for s in per_sample])),
            "mean_Linf": float(np.mean([s["Linf"] for s in per_sample])),
            "mean_relative_L2": float(np.mean([s["relative_L2"] for s in per_sample])),
            "mean_wall_s": enc_wall_total / args.num_samples,
        },
        "per_sample": per_sample,
    }

    out_path = (
        ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.block}_validation.json"
    )
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print("\n" + "=" * 60)
    print(f"  {cfg['short']} {args.block.upper()} block (N={args.num_samples})")
    print("=" * 60)
    print(f"  mean L1            : {m['mean_L1']:.6f}")
    print(f"  mean L2            : {m['mean_L2']:.6f}")
    print(f"  mean Linf          : {m['mean_Linf']:.6f}")
    print(f"  mean relative L2   : {100 * m['mean_relative_L2']:.3f}%")
    print(f"  mean wall / sample : {m['mean_wall_s']:.2f}s")
    print(f"\n  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
