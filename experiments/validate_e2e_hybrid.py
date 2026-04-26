#!/usr/bin/env python3
"""LPAN-Hybrid validation: end-to-end CKKS + PBRP checkpoints vs plaintext.

Mirrors :mod:`experiments.validate_e2e_openfhe` but consults
``CHECKPOINT_SCHEDULES`` from ``fhe_thesis.config`` and replaces
selected bootstrap calls with the Polynomial-Bounded Re-Encryption
Protocol (PBRP), implemented in
:mod:`fhe_thesis.encryption.checkpoint`.

Usage
-----
    python experiments/validate_e2e_hybrid.py \
        --model tiny --task sst2 --num-samples 1 \
        --seq-len 4 --mult-depth 45 --ring-dim 16384 --k 1

``--k 0`` reproduces the pure-FHE baseline (no checkpoints).
``--k 1`` inserts one checkpoint per encoder layer (after FFN).
``--k 2`` inserts checkpoints both mid-layer and at end.

Writes ``results/encrypted_inference/<model>_<task>_hybrid_k<k>.json``.
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
    CHECKPOINT_SCHEDULES,
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
    p.add_argument("--mult-depth", type=int, default=45)
    p.add_argument("--ring-dim", type=int, default=1 << 14)
    p.add_argument("--checkpoint", default=None,
                   help="Path to LPAN best_model dir (else autodiscover Stage-4)")
    p.add_argument("--k", type=int, default=1, choices=(0, 1, 2),
                   help="Checkpoint density: 0=pure FHE, 1=once/layer, 2=mid+end/layer")
    return p.parse_args()


def _resolve_ckpt(model_key: str, task: str, override: str | None) -> Path:
    if override:
        return Path(override)
    # Stage-4 first, fall back to Stage-3.
    flat = MULTI_MODEL_DIR / model_key / "stage4_range_aware" / "best_model"
    nested = MULTI_MODEL_DIR / model_key / task / "stage4_range_aware" / "best_model"
    flat3 = MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"
    nested3 = MULTI_MODEL_DIR / model_key / task / "staged_lpan_final" / "best_model"
    for cand in (nested, flat, nested3, flat3):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No LPAN checkpoint for {model_key}/{task}")


def _sync_intervals(layer_coeffs, bert_layer):
    g = bert_layer.intermediate.intermediate_act_fn
    layer_coeffs["GELU"] = _dc_replace(layer_coeffs["GELU"],
                                        interval=(float(g.a), float(g.b)))
    ln = bert_layer.output.LayerNorm
    layer_coeffs["LN"] = _dc_replace(layer_coeffs["LN"],
                                      interval=(float(ln.a), float(ln.b)))
    if "Softmax" in layer_coeffs:
        sm = bert_layer.attention.self.poly_softmax
        layer_coeffs["Softmax"] = _dc_replace(layer_coeffs["Softmax"],
                                               interval=(float(sm.a), float(sm.b)))


# ── task plumbing ──────────────────────────────────────────────────────


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


# ── main ───────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    hidden = cfg["hidden"]
    schedule = CHECKPOINT_SCHEDULES[args.model][args.k]
    schedule_set = set((li, pos) for (li, pos) in schedule)

    print(
        f"[hybrid] model={args.model} ({cfg['short']}) hidden={hidden} "
        f"layers={cfg['layers']} heads={cfg['heads']}  task={args.task}  "
        f"k={args.k}  seq_len={args.seq_len}"
    )
    print(f"[hybrid] schedule (k={args.k}): {schedule}")

    ckpt = _resolve_ckpt(args.model, args.task, args.checkpoint)
    print(f"[hybrid] ckpt = {ckpt}")

    # ── heavy imports ──────────────────────────────────────────────
    import torch as _torch
    from transformers import AutoTokenizer

    from fhe_thesis.encryption.checkpoint import (
        CheckpointSession,
        identity,
        reencrypt_checkpoint,
    )
    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    from fhe_thesis.encryption.ops import enc_linear
    from fhe_thesis.encryption.packing import TokenPackedTensor
    from fhe_thesis.encryption.protocol import (
        encrypt_attention_block,
        encrypt_ffn_block,
        load_model_weights,
    )
    from fhe_thesis.models.lpan_loader import load_lpan_model

    # ── load model + weights + coefficients ─────────────────────────
    print("\n[1/4] loading LPAN model + extracted coefficients ...")
    model = load_lpan_model(args.model, ckpt, num_labels=2)
    weights = load_model_weights(args.model, checkpoint_path=str(ckpt))
    coeffs = load_coefficients(args.model, task=args.task)
    for li in range(len(weights.layers)):
        _sync_intervals(coeffs[li], model.bert.encoder.layer[li])
    print(f"      layers={len(weights.layers)}  cls_W={weights.cls_W is not None}")

    # ── slot + backend ─────────────────────────────────────────────
    ffn_inter = 4 * hidden
    needed = max(hidden, ffn_inter, args.seq_len, args.seq_len * args.seq_len)
    num_slots = 1
    while num_slots < needed:
        num_slots <<= 1
    print(f"      num_slots = {num_slots}")

    print("\n[2/4] booting OpenFHE backend ...")
    import openfhe as ofhe
    t0 = time.perf_counter()
    backend = OpenFHEBackend(
        multiplicative_depth=args.mult_depth,
        ring_dim=args.ring_dim,
        scaling_mod_size=59, first_mod_size=60,
        enable_bootstrap=True, num_slots=num_slots,
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

    # ── observe layer-output intervals once for PBRP mask calibration ─
    # Use the LayerNorm 'b' (the post-LN output bound) as the hidden-state
    # interval; since post-LN values stay in roughly the same range, the
    # mask scale is small relative to slot precision.
    layer_intervals: dict[int, tuple[float, float]] = {}
    for li in range(len(weights.layers)):
        ln_b = float(model.bert.encoder.layer[li].output.LayerNorm.b)
        # LN outputs are roughly zero-centred but can dip negative;
        # use symmetric interval to be safe.
        layer_intervals[li] = (-ln_b, ln_b)

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
        sess = CheckpointSession()
        t_sample = time.perf_counter()
        h = TokenPackedTensor.encrypt(backend, embeds)

        for li, layer in enumerate(weights.layers):
            t_layer = time.perf_counter()
            chk_mid = (li, "mid") in schedule_set
            chk_end = (li, "end") in schedule_set

            # Attention block
            h, _ = encrypt_attention_block(backend, h, layer, coeffs[li], weights.num_heads)

            # Mid-layer refresh: PBRP if scheduled, else bootstrap
            if chk_mid:
                new_cts = [
                    reencrypt_checkpoint(
                        backend, ct, layer_intervals[li], identity,
                        layer_idx=li, position="mid",
                        n_active_slots=hidden, session=sess,
                    )
                    for ct in h.cts
                ]
            else:
                new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)

            # FFN block
            h, _ = encrypt_ffn_block(backend, h, layer, coeffs[li])

            # End-of-layer refresh
            if chk_end:
                new_cts = [
                    reencrypt_checkpoint(
                        backend, ct, layer_intervals[li], identity,
                        layer_idx=li, position="end",
                        n_active_slots=hidden, session=sess,
                    )
                    for ct in h.cts
                ]
            else:
                new_cts = [backend.bootstrap(ct) for ct in h.cts]
            h = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=h.hidden_dim)

            try:
                cls_dec = backend.decrypt(h.cts[0])[: h.hidden_dim]
                cls_mag = float(np.max(np.abs(cls_dec)))
            except Exception as e:
                cls_mag = float("nan")
                print(f"     layer {li}: DECRYPT FAILED: {e}")
            print(
                f"     layer {li}: {time.perf_counter() - t_layer:6.1f}s  "
                f"level={backend.get_level(h.cts[0])}/{args.mult_depth}  "
                f"|CLS|max={cls_mag:.3f}  "
                f"chk[mid={chk_mid},end={chk_end}]"
            )

        # Classifier head
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        try:
            enc_logits = np.asarray(out_ct.decrypt(backend))[0].astype(np.float64)
        except Exception as e:
            print(f"     direct cls decrypt failed ({e}); bootstrapping then retry")
            new_cts = [backend.bootstrap(ct) for ct in out_ct.cts]
            out_ct = TokenPackedTensor.from_ciphertexts(new_cts, hidden_dim=out_ct.hidden_dim)
            enc_logits = np.asarray(out_ct.decrypt(backend))[0].astype(np.float64)
        sample_wall = time.perf_counter() - t_sample

        diff = pt_out - enc_logits
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        linf = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(pt_out), 1e-12))
        enc_pred = int(np.argmax(enc_logits))
        argmax_agree = enc_pred == pt_pred

        chk_summary = sess.summary()
        per_sample.append({
            "ds_idx": int(ds_idx),
            "true_label": true_label,
            "pt_logits": pt_out.tolist(),
            "enc_logits": enc_logits.tolist(),
            "pt_pred": pt_pred, "enc_pred": enc_pred,
            "argmax_agree": argmax_agree,
            "L1": l1, "L2": l2, "Linf": linf, "relative_L2": rel,
            "wall_s": sample_wall,
            "checkpoint_summary": chk_summary,
        })
        print(
            f"     pt_logits={pt_out.round(4).tolist()}  "
            f"enc_logits={enc_logits.round(4).tolist()}\n"
            f"     L1={l1:.4f}  L2={l2:.4f}  Linf={linf:.4f}  "
            f"relL2={100 * rel:.2f}%  argmax_match={argmax_agree}  "
            f"({sample_wall:.1f}s, chkpts={chk_summary['checkpoints']}, "
            f"chk_total={chk_summary['total_ms']:.1f}ms)"
        )

    arr = lambda k: np.asarray([s[k] for s in per_sample], dtype=np.float64)
    summary = {
        "model": args.model, "task": args.task, "k": args.k,
        "schedule": schedule,
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

    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_{args.task}_hybrid_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    m = summary["metrics"]
    print("\n" + "=" * 60)
    print(f"  Hybrid validation: {cfg['short']} on {args.task.upper()}  "
          f"k={args.k} (N={args.num_samples})")
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
