#!/usr/bin/env python3
"""Decrypt + validate PF-SR encrypted inference accuracy.

Loads an LPAN-trained checkpoint, runs both the plaintext forward
pass and the PF-SR encrypted pipeline on the same validation slice,
then dumps a JSON record with:

* ``accuracy_plain``     — plaintext LPAN accuracy on the slice
* ``accuracy_decrypted`` — decrypted PF-SR accuracy on the slice
* ``agreement``          — fraction of inputs where argmax matches
* ``mean_abs_logit_delta`` — mean ‖y_plain − y_decrypted‖₁ / C
* ``per_sample``         — list of per-input plaintext / decrypted logits

Usage
-----
    python experiments/validate_encrypted_accuracy.py \
        --model {tiny|mini|small|base} \
        --task {sst2|mrpc|qnli|qqp} \
        [--num-samples 100] [--seq-len 8] [--checkpoint <path>]

Heavy compute: requires TenSEAL with N=32768 for any non-Tiny model.
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
from fhe_thesis.tasks import GLUE_TASKS, get_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--task", required=True, choices=sorted(GLUE_TASKS))
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Truncate inputs to this many tokens (PF-SR seq length).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="LPAN checkpoint dir. Default: results/multi_model/<model>/staged_lpan_final/best_model",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _default_checkpoint(model_key: str) -> Path:
    return MULTI_MODEL_DIR / model_key / "staged_lpan_final" / "best_model"


def main() -> int:
    args = parse_args()
    ensure_dirs()

    cfg = MODEL_REGISTRY[args.model]
    task = get_task(args.task)
    ckpt = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args.model)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"  → run: python run_staged_lpan.py --model {args.model} --task {args.task}"
        )

    print(f"[validate] model = {args.model} ({cfg['short']})")
    print(f"[validate] task  = {args.task} ({task.description})")
    print(f"[validate] ckpt  = {ckpt}")
    print(f"[validate] N     = {args.num_samples} samples, seq_len = {args.seq_len}")

    # Heavy imports
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from fhe_thesis.encryption import TenSEALBackend
    from fhe_thesis.encryption.context import make_context
    from fhe_thesis.encryption.depth import transformer_layer_depth
    from fhe_thesis.encryption.protocol import run_phase

    # ── Load plaintext LPAN model ──────────────────────────────────────
    print("\n[1/4] Loading plaintext LPAN checkpoint …")
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt), num_labels=task.num_labels
    ).eval()

    # ── Load validation slice ──────────────────────────────────────────
    print(f"[2/4] Loading {args.task} validation slice …")
    ds = load_dataset("glue", task.name, split="validation")
    n = min(args.num_samples, len(ds))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist()
    ds = ds.select(indices)

    # ── Boot CKKS context (≥ 23 L_enc + 2 levels for full model) ───────
    layer_depth = transformer_layer_depth()
    mult_depth = layer_depth * cfg["layers"] + 2
    print(f"[3/4] Booting CKKS context (mult_depth = {mult_depth} levels) …")
    t0 = time.time()
    ctx = make_context(mult_depth=mult_depth)
    backend = TenSEALBackend(ctx)
    print(
        f"      boot wall = {time.time() - t0:.1f}s, "
        f"capabilities: {backend.capabilities}"
    )

    # ── Per-sample loop ────────────────────────────────────────────────
    print(f"\n[4/4] Running {n} encrypted inferences …")
    plain_correct = 0
    dec_correct = 0
    agree = 0
    deltas: list[float] = []
    per_sample: list[dict] = []
    enc_wall_total = 0.0

    for i, ex in enumerate(ds):
        # tokenise (1 or 2 text fields depending on the task)
        text_args = [ex[f] for f in task.text_fields]
        enc_inputs = tok(
            *text_args,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=args.seq_len,
        )

        # plaintext forward (returns logits + uses LPAN polys + plaintext arithmetic)
        with torch.no_grad():
            plain_logits = model(**enc_inputs).logits[0].numpy()
            # Embeddings for the encrypted side: use the same embeddings the
            # plaintext model just consumed, truncated to seq_len.
            embeds = model.bert.embeddings(enc_inputs["input_ids"]).detach()[0].numpy()

        embeds = embeds[: args.seq_len].astype(np.float64)

        # encrypted forward (returns numpy logits — decryption already inside)
        t1 = time.time()
        enc_logits, _ = run_phase(
            "model",
            args.model,
            backend,
            embeds,
            checkpoint_path=str(ckpt),
        )
        enc_wall_total += time.time() - t1
        enc_logits = np.asarray(enc_logits).reshape(-1)[: task.num_labels]

        label = int(ex[task.label_column])
        plain_pred = int(np.argmax(plain_logits))
        dec_pred = int(np.argmax(enc_logits))

        plain_correct += int(plain_pred == label)
        dec_correct += int(dec_pred == label)
        agree += int(plain_pred == dec_pred)
        deltas.append(float(np.mean(np.abs(plain_logits - enc_logits))))

        per_sample.append(
            {
                "index": int(indices[i]),
                "label": label,
                "plain_logits": plain_logits.tolist(),
                "decrypted_logits": enc_logits.tolist(),
                "plain_pred": plain_pred,
                "decrypted_pred": dec_pred,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(
                f"  [{i + 1:>4d}/{n}] "
                f"plain={plain_correct / (i + 1):.3f}  "
                f"dec={dec_correct / (i + 1):.3f}  "
                f"agree={agree / (i + 1):.3f}  "
                f"|Δlogit|={np.mean(deltas):.4f}"
            )

    summary = {
        "model": args.model,
        "model_short": cfg["short"],
        "task": args.task,
        "num_samples": n,
        "seq_len": args.seq_len,
        "checkpoint": str(ckpt),
        "ckks": {
            "mult_depth": mult_depth,
            "layer_depth": layer_depth,
            "num_layers": cfg["layers"],
        },
        "accuracy_plain": plain_correct / n,
        "accuracy_decrypted": dec_correct / n,
        "agreement": agree / n,
        "mean_abs_logit_delta": float(np.mean(deltas)),
        "encrypted_wall_total_s": enc_wall_total,
        "encrypted_wall_per_sample_s": enc_wall_total / n,
        "per_sample": per_sample,
    }

    out_path = ENCRYPTED_INFERENCE_DIR / f"{args.model}_validation_{args.task}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  {cfg['short']} on {args.task.upper()} ({n} samples)")
    print("=" * 70)
    print(f"  plaintext LPAN accuracy : {summary['accuracy_plain']:.4f}")
    print(f"  decrypted   PF-SR accuracy: {summary['accuracy_decrypted']:.4f}")
    print(f"  argmax agreement          : {summary['agreement']:.4f}")
    print(f"  mean |Δlogit|             : {summary['mean_abs_logit_delta']:.4f}")
    print(
        f"  encrypted wall / sample   : {summary['encrypted_wall_per_sample_s']:.2f}s"
    )
    print(f"\n  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
