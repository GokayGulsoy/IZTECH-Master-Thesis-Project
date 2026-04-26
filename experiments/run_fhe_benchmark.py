"""FHE inference benchmark — Phase H of the optimization plan.

Runs the LPAN-FHE pipeline at each optimization level and records
wall-clock latency per sample, accuracy vs plaintext, and per-op
timing breakdown. Results are saved to
``results/benchmarks/fhe_benchmark_{model}_{task}.json``.

Usage (on the 32-vCPU CPU pod after deploying OpenFHE + HEXL):
    python experiments/run_fhe_benchmark.py \\
        --model base --task sst2 \\
        --n-samples 100 \\
        --max-seq-len 64 \\
        --n-jobs 32 \\
        --checkpoint results/multi_model/sst2/base/staged_lpan_final/best_model \\
        --out results/benchmarks/

For a local smoke-test without OpenFHE installed, use --dry-run to run
only the plaintext accuracy check (skips encryption).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


# ── Argument parsing ────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="LPAN-FHE latency benchmark")
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "mini", "small", "base"],
                   help="BERT variant (default: tiny)")
    p.add_argument("--task", default="sst2", choices=["sst2", "mrpc", "qnli"],
                   help="GLUE task (default: sst2)")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Number of validation samples to benchmark (default: 20)")
    p.add_argument("--max-seq-len", type=int, default=64,
                   help="Truncate sequences to this length before encryption (O4, default: 64)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Token-level parallelism threads (O5, default: 1; -1=all CPUs)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to LPAN-trained checkpoint directory (default: pretrained HF weights)")
    p.add_argument("--out", type=str, default="results/benchmarks",
                   help="Output directory for JSON results")
    p.add_argument("--mult-depth", type=int, default=25,
                   help="CKKS multiplicative depth (default: 25)")
    p.add_argument("--ring-dim", type=int, default=1 << 16,
                   help="CKKS ring dimension N (default: 65536)")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Disable bootstrapping (faster setup, limits depth)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip encryption, only check plaintext accuracy")
    p.add_argument("--phase", default="model",
                   choices=["ffn", "attention", "layer", "model"],
                   help="Which protocol phase to benchmark (default: model)")
    return p.parse_args()


# ── Dataset loading ─────────────────────────────────────────────────────

def _load_dataset(task: str, n_samples: int, max_seq_len: int):
    """Load validation set embeddings via HuggingFace datasets + tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    hf_name = {"sst2": "glue/sst2", "mrpc": "glue/mrpc", "qnli": "glue/qnli"}[task]
    text_col = {"sst2": "sentence", "mrpc": "sentence1", "qnli": "question"}[task]
    label_col = "label"

    ds = load_dataset("glue", task, split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))

    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    samples = []
    for item in ds:
        enc = tokenizer(
            item[text_col],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        samples.append({
            "input_ids": enc["input_ids"][0].numpy(),
            "attention_mask": enc["attention_mask"][0].numpy(),
            "label": int(item[label_col]),
        })
    return samples


def _get_embeddings(model_key: str, samples: list, checkpoint_path: str | None):
    """Run the embedding layer (plaintext) and return numpy activations."""
    import torch
    from transformers import AutoModelForSequenceClassification

    from fhe_thesis.config import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)
    model.eval()

    embeddings, labels = [], []
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor(s["input_ids"]).unsqueeze(0)
            mask = torch.tensor(s["attention_mask"]).unsqueeze(0)
            emb = model.bert.embeddings(ids)  # (1, seq_len, hidden)
            embeddings.append(emb.squeeze(0).numpy())
            labels.append(s["label"])
    return embeddings, labels


# ── Plaintext accuracy baseline ─────────────────────────────────────────

def _plaintext_accuracy(model_key: str, samples: list, checkpoint_path: str | None) -> float:
    """Run the full plaintext (non-encrypted) model and return accuracy."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from fhe_thesis.config import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)
    model.eval()

    correct = 0
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor(s["input_ids"]).unsqueeze(0)
            mask = torch.tensor(s["attention_mask"]).unsqueeze(0)
            logits = model(ids, attention_mask=mask).logits
            pred = int(logits.argmax(-1).item())
            if pred == s["label"]:
                correct += 1
    return correct / len(samples)


# ── Per-sample FHE inference ─────────────────────────────────────────────

def _run_fhe_sample(backend, weights, coeffs, emb_np, max_seq_len, n_jobs):
    """Encrypt, infer, decrypt one sample. Returns (logits, timings)."""
    from fhe_thesis.encryption.protocol import encrypt_inference
    logits, timings = encrypt_inference(
        backend, emb_np, weights, coeffs,
        max_seq_len=max_seq_len, n_jobs=n_jobs,
    )
    return logits, timings


# ── Main benchmark loop ──────────────────────────────────────────────────

def main():
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LPAN-FHE Benchmark ===")
    print(f"  model={args.model}  task={args.task}  n_samples={args.n_samples}")
    print(f"  max_seq_len={args.max_seq_len}  n_jobs={args.n_jobs}")
    print(f"  dry_run={args.dry_run}  no_bootstrap={args.no_bootstrap}\n")

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading validation samples...")
    samples = _load_dataset(args.task, args.n_samples, args.max_seq_len)

    # ── Plaintext accuracy ─────────────────────────────────────────────
    print("Computing plaintext accuracy baseline...")
    plain_acc = _plaintext_accuracy(args.model, samples, args.checkpoint)
    print(f"  Plaintext accuracy: {plain_acc*100:.2f}%")

    results: Dict = {
        "model": args.model,
        "task": args.task,
        "n_samples": len(samples),
        "max_seq_len": args.max_seq_len,
        "n_jobs": args.n_jobs,
        "mult_depth": args.mult_depth,
        "ring_dim": args.ring_dim,
        "plaintext_accuracy": plain_acc,
        "fhe_samples": [],
    }

    if args.dry_run:
        print("--dry-run: skipping FHE encryption. Done.")
        _save_results(results, out_dir, args.model, args.task)
        return

    # ── Build OpenFHE backend ─────────────────────────────────────────
    print("Initialising OpenFHE backend (this takes ~1–3 min for key generation)...")
    t0 = time.time()
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    backend = OpenFHEBackend(
        multiplicative_depth=args.mult_depth,
        ring_dim=args.ring_dim,
        enable_bootstrap=not args.no_bootstrap,
        num_threads=args.n_jobs if args.n_jobs > 0 else os.cpu_count(),
    )
    keygen_time = time.time() - t0
    print(f"  Key generation: {keygen_time:.1f}s")
    results["keygen_time_s"] = keygen_time

    # ── Load weights + coefficients ───────────────────────────────────
    print("Loading model weights and polynomial coefficients...")
    from fhe_thesis.encryption.protocol import load_model_weights
    from fhe_thesis.encryption.coefficients import load_coefficients
    weights = load_model_weights(args.model, checkpoint_path=args.checkpoint)
    coeffs = load_coefficients(args.model, task=args.task)

    # Get embeddings (plaintext embedding layer runs outside FHE)
    print("Computing plaintext embeddings for all samples...")
    embeddings, labels = _get_embeddings(args.model, samples, args.checkpoint)

    # ── FHE inference loop ────────────────────────────────────────────
    print(f"\nRunning FHE inference on {len(samples)} samples...")
    fhe_correct = 0
    all_latencies: List[float] = []
    all_timings: List[Dict] = []

    for idx, (emb, label) in enumerate(zip(embeddings, labels)):
        t_start = time.time()
        logits, timings = _run_fhe_sample(
            backend, weights, coeffs, emb, args.max_seq_len, args.n_jobs
        )
        wall = time.time() - t_start

        pred = int(np.argmax(logits[0]))  # CLS token
        correct = pred == label
        fhe_correct += int(correct)
        all_latencies.append(wall)
        all_timings.append(timings)

        results["fhe_samples"].append({
            "idx": idx,
            "label": label,
            "pred": pred,
            "correct": correct,
            "wall_time_s": wall,
            "timings": timings,
        })

        print(f"  [{idx+1:3d}/{len(samples)}]  label={label} pred={pred} "
              f"{'✓' if correct else '✗'}  {wall:.1f}s")

    fhe_acc = fhe_correct / len(samples)
    mean_lat = float(np.mean(all_latencies))
    median_lat = float(np.median(all_latencies))

    results["fhe_accuracy"] = fhe_acc
    results["agreement"] = fhe_correct / len(samples)
    results["mean_latency_s"] = mean_lat
    results["median_latency_s"] = median_lat
    results["min_latency_s"] = float(np.min(all_latencies))
    results["max_latency_s"] = float(np.max(all_latencies))

    # Per-op aggregated timings
    if all_timings:
        op_keys = [k for k in all_timings[0] if k != "total"]
        results["mean_op_timings_s"] = {
            k: float(np.mean([t.get(k, 0) for t in all_timings]))
            for k in op_keys
        }

    print(f"\n=== Results ===")
    print(f"  FHE accuracy:      {fhe_acc*100:.2f}%  (plaintext: {plain_acc*100:.2f}%)")
    print(f"  Mean latency:      {mean_lat:.2f}s/sample")
    print(f"  Median latency:    {median_lat:.2f}s/sample")
    print(f"  Amortized (B=42):  {mean_lat/42:.3f}s/sample")

    _save_results(results, out_dir, args.model, args.task)


def _save_results(results: Dict, out_dir: Path, model: str, task: str):
    fname = out_dir / f"fhe_benchmark_{model}_{task}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {fname}")


if __name__ == "__main__":
    main()
