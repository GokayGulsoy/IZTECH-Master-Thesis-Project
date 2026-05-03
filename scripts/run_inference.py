"""Single-sentence FHE inference entrypoint.

Given a text string, tokenizes it, runs the LPAN-FHE pipeline, and
prints the predicted label.

Usage:
    python scripts/run_inference.py \\
        --task sst2 \\
        --text "This movie was absolutely fantastic!" \\
        --model base \\
        --checkpoint results/multi_model/sst2/base/staged_lpan_final/best_model \\
        --max-seq-len 64 \\
        --n-jobs 8

For MRPC (sentence-pair task):
    python scripts/run_inference.py \\
        --task mrpc \\
        --text "A dog ran across the lawn." \\
        --text2 "A canine sprinted over the grass." \\
        --model base
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="LPAN-FHE single-sentence inference")
    p.add_argument("--task", default="sst2", choices=["sst2", "mrpc", "qnli"])
    p.add_argument("--text", required=True, help="Input sentence (or question for QNLI)")
    p.add_argument("--text2", default=None,
                   help="Second sentence (required for MRPC / QNLI)")
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "mini", "small", "base"],
                   help="BERT variant (default: tiny)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to LPAN checkpoint directory")
    p.add_argument("--max-seq-len", type=int, default=64,
                   help="Truncate to this many tokens before encryption (O4)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Token-level parallelism (O5; -1=all CPUs)")
    p.add_argument("--mult-depth", type=int, default=25)
    p.add_argument("--ring-dim", type=int, default=1 << 16)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--no-fhe", action="store_true",
                   help="Run plaintext LPAN model (skips encryption, for comparison)")
    p.add_argument("--backend", default="openfhe", choices=["openfhe", "heongpu"],
                   help="CKKS backend (heongpu requires the H100 wrapper)")
    p.add_argument("--layout", default="token", choices=["token", "matrix"],
                   help="ciphertext packing: token (1 ct/token) or matrix (B tokens/ct)")
    p.add_argument("--block", type=int, default=0,
                   help="matrix-pack block (0 = auto = next_pow2(max_dim))")
    p.add_argument("--poly-degree", type=int, default=8,
                   help="Polynomial degree for GELU/LN/Softmax fits when no "
                        "trained checkpoint is found (lower = shallower depth, "
                        "fewer ciphertext multiplies, less bootstrap pressure)")
    return p.parse_args()


_LABEL_MAPS = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not-paraphrase", 1: "paraphrase"},
    "qnli": {0: "entailment", 1: "not-entailment"},
}


def _tokenize(task, text, text2, max_seq_len, model_key):
    from transformers import AutoTokenizer
    from fhe_thesis.config import MODEL_REGISTRY

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if text2:
        enc = tokenizer(text, text2,
                        max_length=max_seq_len, padding="max_length",
                        truncation=True, return_tensors="pt")
    else:
        enc = tokenizer(text,
                        max_length=max_seq_len, padding="max_length",
                        truncation=True, return_tensors="pt")
    return enc


def _get_embedding(enc, model_key, checkpoint_path):
    import torch
    from transformers import AutoModelForSequenceClassification
    from fhe_thesis.config import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)
    model.eval()
    with torch.no_grad():
        emb = model.bert.embeddings(enc["input_ids"])  # (1, L, H)
        logits_plain = model(**enc).logits
    return emb.squeeze(0).numpy(), int(logits_plain.argmax(-1).item()), model


def _plaintext_inference(enc, model):
    import torch
    with torch.no_grad():
        logits = model(**enc).logits
    return int(logits.argmax(-1).item())


def main():
    args = _parse_args()
    label_map = _LABEL_MAPS[args.task]

    print(f"\n--- LPAN-FHE Inference [{args.task.upper()}] ---")
    print(f"  Input: {args.text!r}")
    if args.text2:
        print(f"  Input2: {args.text2!r}")
    print(f"  Model: bert-{args.model}, max_seq_len={args.max_seq_len}")

    # Tokenize
    enc = _tokenize(args.task, args.text, args.text2, args.max_seq_len, args.model)
    actual_tokens = int(enc["attention_mask"].sum().item())
    print(f"  Tokens (non-pad): {actual_tokens}")

    # Get plaintext embedding + plain prediction
    emb_np, plain_pred, pt_model = _get_embedding(enc, args.model, args.checkpoint)
    print(f"  Plaintext prediction: {label_map[plain_pred]!r} (label {plain_pred})")

    if args.no_fhe:
        print("\n--no-fhe: done (plaintext only).")
        return

    # Init FHE backend
    t0 = time.time()
    if args.backend == "heongpu":
        print("\nInitialising HEonGPU backend (H100, bootstrap-capable)...")
        from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
        # Bootstrap-capable chain matches the HEonGPU example:
        # 31 Q primes (60 + 50*30), 3 P primes (60*3), scale=2^50,
        # secret hamming weight 16. sec_none=True is required for this
        # ring/chain combo at N=2^16; the secure baseline (sec_none=False)
        # would require shorter chain + more frequent bootstraps.
        backend = HEonGPUBackend(
            poly_modulus_degree=1 << 16,
            q_prime_bits=(60,) + (50,) * 30,
            p_prime_bits=(60, 60, 60),
            scale_bits=50,
            bootstrap_hamming_weight=16,
            sec_none=True,
        )
        if not args.no_bootstrap:
            print("  Configuring bootstrapping (this rebuilds Galois keys)...")
            backend.configure_bootstrapping()
    else:
        print("\nInitialising OpenFHE backend...")
        from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
        backend = OpenFHEBackend(
            multiplicative_depth=args.mult_depth,
            ring_dim=args.ring_dim,
            enable_bootstrap=not args.no_bootstrap,
            num_threads=args.n_jobs if args.n_jobs > 0 else os.cpu_count(),
        )
    print(f"  Key generation: {time.time()-t0:.1f}s")

    # Load weights + coefficients
    from fhe_thesis.encryption.protocol import load_model_weights
    from fhe_thesis.encryption.coefficients import load_coefficients
    weights = load_model_weights(args.model, checkpoint_path=args.checkpoint)
    coeffs = load_coefficients(args.model, task=args.task, degree=args.poly_degree)

    # FHE inference
    print(f"\nRunning encrypted inference (layout={args.layout})...")
    t0 = time.time()
    if args.layout == "matrix":
        from fhe_thesis.encryption.protocol import encrypt_inference_matrix
        logits, timings = encrypt_inference_matrix(
            backend, emb_np, weights, coeffs,
            max_seq_len=args.max_seq_len,
            block=args.block,
        )
    else:
        from fhe_thesis.encryption.protocol import encrypt_inference
        logits, timings = encrypt_inference(
            backend, emb_np, weights, coeffs,
            max_seq_len=args.max_seq_len,
            n_jobs=args.n_jobs,
        )
    wall = time.time() - t0

    # Classify using CLS token output (index 0)
    fhe_pred = int(np.argmax(logits[0]))
    print(f"\n  FHE prediction:  {label_map[fhe_pred]!r} (label {fhe_pred})")
    print(f"  Wall-clock time: {wall:.2f}s")
    print(f"  Agree w/ plain:  {'yes' if fhe_pred == plain_pred else 'NO'}")

    print("\n  Per-op breakdown:")
    for op, t in sorted(timings.items()):
        print(f"    {op:35s} {t:.3f}s")


if __name__ == "__main__":
    main()
