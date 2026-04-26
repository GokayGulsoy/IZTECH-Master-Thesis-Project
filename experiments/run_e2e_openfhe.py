#!/usr/bin/env python3
"""End-to-end encrypted BERT inference using OpenFHE with bootstrapping.

Loads a Stage-4 LPAN-trained checkpoint, runs the full encoder + classifier
under FHE, and bootstraps every ciphertext between encoder layers so that
a single ~20-level CKKS budget is enough for any model depth.

Usage
-----
    python experiments/run_e2e_openfhe.py --model tiny --task sst2 --seq-len 8

Outputs a JSON report to ``results/encrypted_inference/<model>_<task>_e2e.json``.
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
    p.add_argument("--task", default="sst2")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mult-depth",
        type=int,
        default=25,
        help="Mult depth between bootstraps (must cover one encoder layer).",
    )
    p.add_argument(
        "--ring-dim",
        type=int,
        default=1 << 16,
        help="CKKS ring dimension N. 65536 required for 128-bit security at this depth.",
    )
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap (one-block smoke test).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path (default: results/multi_model/<model>/staged_lpan_final/best_model).",
    )
    p.add_argument(
        "--no-classifier",
        action="store_true",
        help="Skip pooler + classifier head (encoder output only).",
    )
    return p.parse_args()


def _resolve_checkpoint(model_key: str, override: str | None) -> str:
    if override:
        return override
    cand = MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"
    if cand.exists():
        return str(cand)
    cand2 = MULTI_MODEL_DIR / model_key / "stage4_range_aware" / "best_model"
    if cand2.exists():
        return str(cand2)
    raise FileNotFoundError(
        f"No LPAN checkpoint found for {model_key} under {MULTI_MODEL_DIR}"
    )


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    hidden = cfg["hidden"]
    print(
        f"[e2e] model={args.model} ({cfg['short']}) "
        f"hidden={hidden} heads={cfg['heads']} layers={cfg['layers']} seq_len={args.seq_len}"
    )

    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    from fhe_thesis.encryption.packing import TokenPackedTensor
    from fhe_thesis.encryption.protocol import (
        encrypt_attention_block,
        encrypt_ffn_block,
        load_model_weights,
    )
    from fhe_thesis.encryption.ops import enc_linear

    ckpt = _resolve_checkpoint(args.model, args.checkpoint)
    print(f"[e2e] checkpoint    = {ckpt}")
    print(f"[e2e] task          = {args.task}")
    print(f"[e2e] mult-depth    = {args.mult_depth}  ring={args.ring_dim}")
    print(f"[e2e] bootstrap     = {not args.no_bootstrap}")

    print("\n[1/4] loading weights & coefficients ...")
    weights = load_model_weights(args.model, checkpoint_path=ckpt)
    coeffs = load_coefficients(args.model, task=args.task)
    print(
        f"      layers={len(weights.layers)}  cls_W={weights.cls_W is not None}"
    )

    # num_slots must hold max(hidden, ffn_intermediate, seq_len*seq_len) so the
    # diagonal-encoded matmul has a valid power-of-two dimension that wraps
    # rotations correctly. FFN intermediate dim = 4·hidden in standard BERT.
    ffn_intermediate = 4 * hidden
    needed_slots = max(hidden, ffn_intermediate, args.seq_len, args.seq_len * args.seq_len)
    # Round up to next power of two
    num_slots = 1
    while num_slots < needed_slots:
        num_slots <<= 1
    print(f"      num_slots     = {num_slots} (covers ffn_inter={ffn_intermediate}, L²={args.seq_len ** 2})")

    print("\n[2/4] booting OpenFHE backend ...")
    import openfhe as ofhe

    t0 = time.perf_counter()
    backend = OpenFHEBackend(
        multiplicative_depth=args.mult_depth,
        ring_dim=args.ring_dim,
        scaling_mod_size=59,
        first_mod_size=60,
        enable_bootstrap=not args.no_bootstrap,
        num_slots=num_slots,
        security_level=ofhe.SecurityLevel.HEStd_128_classic
        if args.ring_dim >= (1 << 16)
        else ofhe.SecurityLevel.HEStd_NotSet,
    )
    setup_time = time.perf_counter() - t0
    print(f"      setup wall    = {setup_time:.1f}s   caps={backend.capabilities}")

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.seq_len, hidden)).astype(np.float64) * 0.1

    print("\n[3/4] encrypting input ...")
    t0 = time.perf_counter()
    h = TokenPackedTensor.encrypt(backend, x)
    enc_time = time.perf_counter() - t0
    print(f"      encrypt wall  = {enc_time:.2f}s   tokens={h.seq_len}")

    timings: dict = {"setup": setup_time, "encrypt": enc_time}
    boot_count = 0
    boot_time_total = 0.0

    print("\n[4/4] encrypted forward ...")
    for i, layer in enumerate(weights.layers):
        t_layer = time.perf_counter()
        h, t_attn = encrypt_attention_block(backend, h, layer, coeffs[i], weights.num_heads)
        # Mid-layer bootstrap: refresh after attention so FFN has full budget.
        if not args.no_bootstrap:
            t_b = time.perf_counter()
            lvl_before = backend.get_level(h.cts[0])
            new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)
            mid_dt = time.perf_counter() - t_b
            boot_count += len(new_cts)
            boot_time_total += mid_dt
            print(
                f"  layer {i} mid-boot: {mid_dt:6.1f}s  ({len(new_cts)} cts)  "
                f"level {lvl_before} → {backend.get_level(h.cts[0])}"
            )
            timings[f"L{i}.bootstrap_mid"] = mid_dt
        h, t_ffn = encrypt_ffn_block(backend, h, layer, coeffs[i])
        layer_time = time.perf_counter() - t_layer
        cur_lvl = backend.get_level(h.cts[0])
        print(
            f"  layer {i}: {layer_time:6.1f}s  "
            f"attn={sum(t_attn.values()):.1f}s ffn={sum(t_ffn.values()):.1f}s  "
            f"level={cur_lvl}/{args.mult_depth}"
        )
        timings[f"L{i}.compute"] = layer_time
        timings[f"L{i}.attn_breakdown"] = t_attn
        timings[f"L{i}.ffn_breakdown"] = t_ffn

        # Insert bootstrap before next layer (skip after last layer if no classifier)
        is_last = i == len(weights.layers) - 1
        need_more_compute = (not is_last) or (not args.no_classifier and weights.cls_W is not None)
        if (not args.no_bootstrap) and need_more_compute:
            t_b = time.perf_counter()
            new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)
            boot_dt = time.perf_counter() - t_b
            boot_count += len(new_cts)
            boot_time_total += boot_dt
            new_lvl = backend.get_level(h.cts[0])
            print(
                f"     bootstrap: {boot_dt:6.1f}s  ({len(new_cts)} cts)  "
                f"level {cur_lvl} → {new_lvl}"
            )
            timings[f"L{i}.bootstrap"] = boot_dt

    if not args.no_classifier and weights.cls_W is not None:
        print("  classifier head ...")
        t0 = time.perf_counter()
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        cls_time = time.perf_counter() - t0
        timings["classifier"] = cls_time
        print(f"      classifier wall = {cls_time:.2f}s")
    else:
        out_ct = h

    print("\n[done] decrypting ...")
    t0 = time.perf_counter()
    out = out_ct.decrypt(backend)
    timings["decrypt"] = time.perf_counter() - t0
    timings["bootstrap_total"] = boot_time_total
    timings["bootstrap_count"] = boot_count
    timings["total_compute"] = (
        sum(v for k, v in timings.items() if k.endswith(".compute"))
        + boot_time_total
        + timings.get("classifier", 0.0)
        + timings["encrypt"]
        + timings["decrypt"]
    )

    print(f"\n  total wall (ex setup) = {timings['total_compute']:.1f}s")
    print(f"  bootstraps            = {boot_count}  ({boot_time_total:.1f}s)")
    print(f"  output shape          = {np.asarray(out).shape}")
    print(f"  output norm           = {float(np.linalg.norm(np.asarray(out))):.4f}")

    report = {
        "model": args.model,
        "task": args.task,
        "checkpoint": ckpt,
        "seq_len": args.seq_len,
        "hidden": hidden,
        "num_layers": len(weights.layers),
        "num_heads": cfg["heads"],
        "ring_dim": args.ring_dim,
        "mult_depth": args.mult_depth,
        "num_slots": num_slots,
        "bootstrap_enabled": not args.no_bootstrap,
        "bootstrap_count": boot_count,
        "bootstrap_total_sec": boot_time_total,
        "setup_sec": setup_time,
        "encrypt_sec": timings["encrypt"],
        "decrypt_sec": timings["decrypt"],
        "total_compute_sec": timings["total_compute"],
        "per_layer_compute_sec": [
            timings[f"L{i}.compute"] for i in range(len(weights.layers))
        ],
        "per_layer_bootstrap_sec": [
            timings.get(f"L{i}.bootstrap", 0.0) for i in range(len(weights.layers))
        ],
        "output_norm": float(np.linalg.norm(np.asarray(out))),
    }
    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.task}_e2e_openfhe.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n→ wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
