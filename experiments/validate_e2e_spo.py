#!/usr/bin/env python3
"""LPAN-SPO validation: Selective Polynomial Outsourcing.

Server holds encrypted weights and runs only **linear ops** under FHE
(Q/K/V projections, attention output projection, FFN-1, FFN-2). At
each layer's checkpoint boundaries the server invokes PBRP: the
client decrypts, applies the polynomial activations and attention
softmax in plaintext, then re-encrypts.

This collapses the per-layer multiplicative depth from ~25 (full
encrypted polynomial evaluation) to ~4 (one mul-plain per linear op +
one rescale). Bootstrap is no longer required, so we drop
``mult_depth`` and skip bootstrap key generation entirely — both setup
and per-op cost shrink dramatically.

Usage
-----
    python experiments/validate_e2e_spo.py \
        --model tiny --task sst2 --num-samples 1 --seq-len 4 \
        --mult-depth 12 --ring-dim 16384

Writes ``results/encrypted_inference/<model>_<task>_spo.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace as _dc_replace
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
    p.add_argument("--task", default="sst2", choices=("sst2", "mrpc", "qnli", "qqp"))
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mult-depth", type=int, default=12,
                   help="Total CKKS depth — 4 linear blocks per layer "
                        "+ checkpoint refreshes mean ~2-4 levels suffice.")
    p.add_argument("--ring-dim", type=int, default=1 << 14)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def _resolve_ckpt(model_key: str, task: str, override: str | None) -> Path:
    if override:
        return Path(override)
    flat = MULTI_MODEL_DIR / model_key / "stage4_range_aware" / "best_model"
    nested = MULTI_MODEL_DIR / model_key / task / "stage4_range_aware" / "best_model"
    flat3 = MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"
    nested3 = MULTI_MODEL_DIR / model_key / task / "staged_lpan_final" / "best_model"
    for cand in (nested, flat, nested3, flat3):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No LPAN checkpoint for {model_key}/{task}")


def _load_task_split(task: str):
    from datasets import load_dataset
    if task == "sst2":
        return load_dataset("glue", "sst2", split="validation"), \
               (lambda r: (r["sentence"], None))
    if task == "mrpc":
        return load_dataset("glue", "mrpc", split="validation"), \
               (lambda r: (r["sentence1"], r["sentence2"]))
    if task == "qnli":
        return load_dataset("glue", "qnli", split="validation"), \
               (lambda r: (r["question"], r["sentence"]))
    if task == "qqp":
        return load_dataset("glue", "qqp", split="validation"), \
               (lambda r: (r["question1"], r["question2"]))
    raise ValueError(f"unknown task {task}")


# ── plaintext activation primitives (run by client at PBRP) ────────────


def pt_softmax_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                          num_heads: int) -> np.ndarray:
    """Multi-head scaled-dot-product attention in plaintext.

    q, k, v are (seq_len, hidden) numpy. Returns (seq_len, hidden).
    """
    seq, hid = q.shape
    head_dim = hid // num_heads
    scale = 1.0 / np.sqrt(head_dim)

    qh = q.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)  # H,S,D
    kh = k.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)

    scores = np.matmul(qh, kh.transpose(0, 2, 1)) * scale  # H,S,S
    # numerically-stable softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.matmul(probs, vh)  # H,S,D
    return out.transpose(1, 0, 2).reshape(seq, hid)


def pt_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def pt_gelu(x: np.ndarray) -> np.ndarray:
    # Exact GELU (the trained polynomial approximates this on a bounded interval).
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                     (x + 0.044715 * x ** 3)))


# ── PBRP pack/unpack helpers ────────────────────────────────────────────


def encrypt_token_tensor(backend, x: np.ndarray):
    """Encrypt (seq_len, hidden) array → list of ciphertexts."""
    from fhe_thesis.encryption.packing import TokenPackedTensor
    return TokenPackedTensor.encrypt(backend, x)


def pbrp_decrypt_unmask(backend, packed, intervals_ab: tuple,
                         rng: np.random.Generator, sess) -> np.ndarray:
    """Server masks each ciphertext with calibrated noise, ships to
    client (here just decrypt + subtract). Returns plaintext (seq, hid).
    """
    from fhe_thesis.encryption.checkpoint import _mask_scale, CheckpointStats

    a, b = intervals_ab
    scale = _mask_scale(a, b)
    n_slots = backend.capabilities.n_slots
    out_rows = np.zeros((packed.seq_len, packed.hidden_dim), dtype=np.float64)

    t_total_dec = 0.0
    for i, ct in enumerate(packed.cts):
        mask = np.zeros(n_slots, dtype=np.float64)
        mask[:packed.hidden_dim] = rng.uniform(0.0, scale, size=packed.hidden_dim)
        ct_masked = backend.add_plain(ct, mask.tolist())
        t0 = time.perf_counter()
        dec = np.asarray(backend.decrypt(ct_masked), dtype=np.float64)[:packed.hidden_dim]
        t_total_dec += time.perf_counter() - t0
        out_rows[i] = dec - mask[:packed.hidden_dim]
    sess["decrypt_ms"] += t_total_dec * 1000
    sess["count"] += 1
    return out_rows


def pbrp_reencrypt(backend, x: np.ndarray, sess) -> object:
    """Client re-encrypts a (seq, hid) plaintext."""
    t0 = time.perf_counter()
    packed = encrypt_token_tensor(backend, x)
    sess["encrypt_ms"] += (time.perf_counter() - t0) * 1000
    return packed


# ── main ───────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    hidden = cfg["hidden"]
    print(
        f"[spo] model={args.model} ({cfg['short']}) hidden={hidden} "
        f"layers={cfg['layers']} heads={cfg['heads']}  task={args.task}  "
        f"seq_len={args.seq_len}  depth={args.mult_depth}"
    )
    ckpt = _resolve_ckpt(args.model, args.task, args.checkpoint)
    print(f"[spo] ckpt = {ckpt}")

    # ── heavy imports ──────────────────────────────────────────────
    import torch as _torch
    from transformers import AutoTokenizer

    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    from fhe_thesis.encryption.ops import enc_linear
    from fhe_thesis.encryption.packing import TokenPackedTensor
    from fhe_thesis.encryption.protocol import load_model_weights
    from fhe_thesis.models.lpan_loader import load_lpan_model

    # ── load model + weights ────────────────────────────────────────
    print("\n[1/4] loading LPAN model + weights ...")
    model = load_lpan_model(args.model, ckpt, num_labels=2)
    weights = load_model_weights(args.model, checkpoint_path=str(ckpt))
    print(f"      layers={len(weights.layers)}  cls_W={weights.cls_W is not None}")

    # ── slot + backend ─────────────────────────────────────────────
    ffn_inter = 4 * hidden
    needed = max(hidden, ffn_inter, args.seq_len)
    num_slots = 1
    while num_slots < needed:
        num_slots <<= 1
    print(f"      num_slots = {num_slots}")

    print("\n[2/4] booting OpenFHE backend (no bootstrap, depth={}) ...".format(args.mult_depth))
    import openfhe as ofhe
    t0 = time.perf_counter()
    backend = OpenFHEBackend(
        multiplicative_depth=args.mult_depth,
        ring_dim=args.ring_dim,
        scaling_mod_size=59, first_mod_size=60,
        enable_bootstrap=False, num_slots=num_slots,
        security_level=ofhe.SecurityLevel.HEStd_128_classic
        if args.ring_dim >= (1 << 16)
        else ofhe.SecurityLevel.HEStd_NotSet,
    )
    setup_time = time.perf_counter() - t0
    print(f"      setup wall = {setup_time:.1f}s")

    # ── load samples ───────────────────────────────────────────────
    print(f"\n[3/4] loading {args.task.upper()} validation samples ...")
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    ds, text_fn = _load_task_split(args.task)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.num_samples, replace=False).tolist()

    # Per-layer LN intervals for mask calibration.
    layer_intervals = []
    for li in range(len(weights.layers)):
        ln1 = model.bert.encoder.layer[li].attention.output.LayerNorm
        ln2 = model.bert.encoder.layer[li].output.LayerNorm
        layer_intervals.append({
            "qkv": (-10.0, 10.0),    # post-linear projections
            "attn_o": (-float(ln1.b), float(ln1.b)),
            "ln1": (-float(ln1.b) * 2, float(ln1.b) * 2),  # post-LN can be larger
            "ffn1": (-15.0, 15.0),    # pre-GELU
            "ffn2": (-float(ln2.b), float(ln2.b)),
            "ln2": (-float(ln2.b) * 2, float(ln2.b) * 2),
        })

    # ── per-sample loop ────────────────────────────────────────────
    per_sample = []
    for s_idx, ds_idx in enumerate(indices):
        row = ds[int(ds_idx)]
        t1, t2 = text_fn(row)
        kw = dict(return_tensors="pt", truncation=True,
                  padding="max_length", max_length=args.seq_len)
        enc_in = tok(t1, t2, **kw) if t2 is not None else tok(t1, **kw)
        true_label = int(row["label"])

        with _torch.no_grad():
            pt_out = model(**enc_in).logits[0].numpy().astype(np.float64)
            embeds = model.bert.embeddings(enc_in["input_ids"])[0].numpy().astype(np.float64)
        pt_pred = int(np.argmax(pt_out))

        print(f"\n  ── sample {s_idx + 1}/{args.num_samples} "
              f"(ds_idx={ds_idx}, true={true_label}) ──")

        sess = {"decrypt_ms": 0.0, "encrypt_ms": 0.0, "count": 0}
        sample_rng = np.random.default_rng(args.seed + s_idx + 1)
        t_sample = time.perf_counter()

        # Initial encryption.
        h_packed = encrypt_token_tensor(backend, embeds)

        for li, layer in enumerate(weights.layers):
            t_layer = time.perf_counter()
            ivl = layer_intervals[li]

            # Save x for residual (in plaintext — known after first PBRP).
            # First layer: we already know embeds. Subsequent: tracked from prev re-encrypt.
            x_plain_residual = h_packed.decrypt(backend) if li > 0 else embeds

            # ── (1) Encrypted Q/K/V projections ───────────────────
            ct_q = enc_linear(backend, h_packed, layer.Wq, layer.bq)
            ct_k = enc_linear(backend, h_packed, layer.Wk, layer.bk)
            ct_v = enc_linear(backend, h_packed, layer.Wv, layer.bv)

            # ── PBRP-1: client decrypts q,k,v → softmax-attention → re-encrypt
            q_pt = pbrp_decrypt_unmask(backend, ct_q, ivl["qkv"], sample_rng, sess)
            k_pt = pbrp_decrypt_unmask(backend, ct_k, ivl["qkv"], sample_rng, sess)
            v_pt = pbrp_decrypt_unmask(backend, ct_v, ivl["qkv"], sample_rng, sess)
            attn_raw = pt_softmax_attention(q_pt, k_pt, v_pt, weights.num_heads)
            h_packed = pbrp_reencrypt(backend, attn_raw, sess)

            # ── (2) Encrypted attention-output projection ────────
            ct_o = enc_linear(backend, h_packed, layer.Wo, layer.bo)

            # ── PBRP-2: client decrypts → residual + LN1 → re-encrypt
            o_pt = pbrp_decrypt_unmask(backend, ct_o, ivl["attn_o"], sample_rng, sess)
            ln1_in = o_pt + x_plain_residual
            ln1_out = pt_layernorm(ln1_in, layer.ln1_gamma, layer.ln1_beta)
            x_for_ffn_residual = ln1_out  # residual for FFN block
            h_packed = pbrp_reencrypt(backend, ln1_out, sess)

            # ── (3) Encrypted FFN-1 ───────────────────────────────
            ct_ffn1 = enc_linear(backend, h_packed, layer.W1, layer.b1)

            # ── PBRP-3: client decrypts → GELU → re-encrypt
            ffn1_pt = pbrp_decrypt_unmask(backend, ct_ffn1, ivl["ffn1"], sample_rng, sess)
            gelu_out = pt_gelu(ffn1_pt)
            h_packed = pbrp_reencrypt(backend, gelu_out, sess)

            # ── (4) Encrypted FFN-2 ───────────────────────────────
            ct_ffn2 = enc_linear(backend, h_packed, layer.W2, layer.b2)

            # ── PBRP-4: client decrypts → residual + LN2 → re-encrypt
            ffn2_pt = pbrp_decrypt_unmask(backend, ct_ffn2, ivl["ffn2"], sample_rng, sess)
            ln2_in = ffn2_pt + x_for_ffn_residual
            ln2_out = pt_layernorm(ln2_in, layer.ln2_gamma, layer.ln2_beta)
            h_packed = pbrp_reencrypt(backend, ln2_out, sess)

            cls_mag = float(np.max(np.abs(ln2_out[0])))
            print(
                f"     layer {li}: {time.perf_counter() - t_layer:6.2f}s  "
                f"|CLS|max={cls_mag:.3f}  pbrp_count={sess['count']}"
            )

        # Classifier head — keep encrypted: pooler then classifier (single linear).
        cls = TokenPackedTensor.from_ciphertexts([h_packed.cts[0]],
                                                  hidden_dim=h_packed.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
            # Pooler uses tanh — apply in plaintext via PBRP.
            pooler_pt = pbrp_decrypt_unmask(backend, cls, ivl["ln2"], sample_rng, sess)
            pooler_pt = np.tanh(pooler_pt)
            cls = pbrp_reencrypt(backend, pooler_pt, sess)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        enc_logits = np.asarray(out_ct.decrypt(backend))[0].astype(np.float64)
        sample_wall = time.perf_counter() - t_sample

        diff = pt_out - enc_logits
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        linf = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(pt_out), 1e-12))
        enc_pred = int(np.argmax(enc_logits))
        argmax_agree = enc_pred == pt_pred

        per_sample.append({
            "ds_idx": int(ds_idx),
            "true_label": true_label,
            "pt_logits": pt_out.tolist(),
            "enc_logits": enc_logits.tolist(),
            "pt_pred": pt_pred, "enc_pred": enc_pred,
            "argmax_agree": argmax_agree,
            "L1": l1, "L2": l2, "Linf": linf, "relative_L2": rel,
            "wall_s": sample_wall,
            "pbrp_count": sess["count"],
            "pbrp_decrypt_ms": sess["decrypt_ms"],
            "pbrp_encrypt_ms": sess["encrypt_ms"],
        })
        print(
            f"     pt_logits={pt_out.round(4).tolist()}  "
            f"enc_logits={enc_logits.round(4).tolist()}\n"
            f"     L1={l1:.4f}  L2={l2:.4f}  Linf={linf:.4f}  "
            f"relL2={100 * rel:.2f}%  argmax_match={argmax_agree}  "
            f"({sample_wall:.1f}s, pbrps={sess['count']}, "
            f"dec={sess['decrypt_ms']:.0f}ms, enc={sess['encrypt_ms']:.0f}ms)"
        )

    arr = lambda k: np.asarray([s[k] for s in per_sample], dtype=np.float64)
    summary = {
        "model": args.model, "task": args.task, "protocol": "SPO",
        "checkpoint": str(ckpt),
        "seq_len": args.seq_len, "num_samples": args.num_samples,
        "ring_dim": args.ring_dim, "mult_depth": args.mult_depth,
        "num_slots": num_slots, "setup_sec": setup_time,
        "metrics": {
            "mean_L1": float(np.mean(arr("L1"))),
            "mean_L2": float(np.mean(arr("L2"))),
            "mean_Linf": float(np.mean(arr("Linf"))),
            "mean_relative_L2": float(np.mean(arr("relative_L2"))),
            "mean_wall_s": float(np.mean(arr("wall_s"))),
            "argmax_agreement": float(np.mean([s["argmax_agree"] for s in per_sample])),
        },
        "per_sample": per_sample,
    }

    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.task}_spo.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print("\n" + "=" * 60)
    print(f"  SPO validation: {cfg['short']} on {args.task.upper()}  (N={args.num_samples})")
    print("=" * 60)
    print(f"  mean L1            : {m['mean_L1']:.6f}")
    print(f"  mean L2            : {m['mean_L2']:.6f}")
    print(f"  mean Linf          : {m['mean_Linf']:.6f}")
    print(f"  mean relative L2   : {100 * m['mean_relative_L2']:.3f}%")
    print(f"  argmax agreement   : {100 * m['argmax_agreement']:.1f}%")
    print(f"  mean wall / sample : {m['mean_wall_s']:.1f}s")
    print(f"\n  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
