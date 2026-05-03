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
    p.add_argument("--task", default="sst2", choices=["sst2", "mrpc", "qnli", "rte"],
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
    p.add_argument("--linear-mixing", action="store_true",
                   help="Use linear mixing model (no attention) instead of standard BERT")
    p.add_argument("--linear-mixing-checkpoint", type=str, default=None,
                   help="Path to linear mixing fine-tuned checkpoint "
                        "(default: results/multi_model/<task>/<model>/linear_mixing_final/best_model)")
    p.add_argument("--hybrid", action="store_true",
                   help="Run HyPER-LPAN hybrid model (LinearMixing + Quad + LPAN composition)")
    p.add_argument("--linear-mixing-layers", type=str, default="0,1,2,3",
                   help="Comma-separated layer indices for LinearMixing (--hybrid only)")
    p.add_argument("--quad-attention-layers", type=str, default="4,5,6,7",
                   help="Comma-separated layer indices for QuadAttention (--hybrid only)")
    p.add_argument("--reduced-degrees", action="store_true",
                   help="Use Phase 2b reduced polynomial degrees per region (--hybrid only)")
    p.add_argument("--word-elimination", default="none",
                   choices=["none", "padding", "content_teacher"],
                   help="FHE-pure word elimination strategy (Ext W). "
                        "'padding' drops [PAD] tokens client-side (lossless, free). "
                        "'content_teacher' uses plaintext layer-0 attention to keep top-k tokens. "
                        "Only applied when --hybrid or --linear-mixing.")
    p.add_argument("--keep-ratio", type=float, default=0.5,
                   help="Fraction of tokens to keep for content_teacher elimination (default: 0.5)")
    p.add_argument("--bootstrap-strategy", default="none",
                   choices=["none", "uniform", "adaptive"],
                   help="Bootstrap insertion strategy. "
                        "'none' (default) = no explicit inter-layer bootstraps. "
                        "Note: depth.py reports each LPAN layer as ~33 levels (worst case) "
                        "vs mult_depth=25, but PS + level-absorption typically halve this. "
                        "Use --measure-depth to log actual ciphertext level consumption. "
                        "'uniform' = bootstrap every --bootstrap-period layers. "
                        "'adaptive' = greedy region-adaptive scheduler (Ext 1).")
    p.add_argument("--bootstrap-period", type=int, default=2,
                   help="For --bootstrap-strategy=uniform: bootstrap every K layers (default: 2)")
    p.add_argument("--bootstrap-budget", type=int, default=22,
                   help="Per-window depth budget after bootstrap (default: 22; budget [3,3] consumes ~12 of 25 levels => 13 left, but PS-actual often lower)")
    p.add_argument("--measure-depth", action="store_true",
                   help="Log ciphertext level consumption per layer (1 sample only) "
                        "to calibrate LAYER_DEPTH constants in depth.py")
    return p.parse_args()


# ── Dataset loading ─────────────────────────────────────────────────────

def _load_dataset(task: str, n_samples: int, max_seq_len: int):
    """Load validation set embeddings via HuggingFace datasets + tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    hf_name = {"sst2": "glue/sst2", "mrpc": "glue/mrpc", "qnli": "glue/qnli", "rte": "glue/rte"}[task]
    text_col = {"sst2": "sentence", "mrpc": "sentence1", "qnli": "question", "rte": "sentence1"}[task]
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


def _get_embeddings(model_key: str, samples: list, checkpoint_path: str | None,
                    need_teacher_scores: bool = False):
    """Run the embedding layer (plaintext) and return numpy activations.

    Returns (embeddings, labels, masks, teacher_scores). teacher_scores is
    None unless need_teacher_scores=True, in which case it contains per-sample
    layer-0 attention CLS-row averaged over heads (used by Ext W content_teacher).
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    from fhe_thesis.config import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=2)
    model.eval()

    embeddings, labels, masks, scores = [], [], [], []
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor(s["input_ids"]).unsqueeze(0)
            mask = torch.tensor(s["attention_mask"]).unsqueeze(0)
            emb = model.bert.embeddings(ids)  # (1, seq_len, hidden)
            embeddings.append(emb.squeeze(0).numpy())
            labels.append(s["label"])
            masks.append(s["attention_mask"].astype(np.int64))
            if need_teacher_scores:
                out = model.bert(ids, attention_mask=mask, output_attentions=True)
                # Layer 0 attentions: (1, heads, seq, seq); use CLS row averaged over heads
                attn0 = out.attentions[0][0].mean(dim=0)[0].cpu().numpy()  # (seq,)
                scores.append(attn0.astype(np.float32))
    return embeddings, labels, masks, (scores if need_teacher_scores else None)


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

def _run_fhe_sample(
    backend,
    weights,
    coeffs,
    emb_np,
    max_seq_len,
    n_jobs,
    phase,
    model,
    checkpoint,
    linear_mixing=False,
    hybrid=False,
    kept_token_indices=None,
    bootstrap_plan=None,
):
    """Encrypt, infer, decrypt one sample. Returns (logits, timings)."""
    if hybrid:
        from fhe_thesis.encryption.protocol import encrypt_inference_hybrid

        logits, timings = encrypt_inference_hybrid(
            backend,
            emb_np,
            weights,
            coeffs,
            max_seq_len=max_seq_len,
            n_jobs=n_jobs,
            kept_token_indices=kept_token_indices,
            bootstrap_plan=bootstrap_plan,
        )
    elif linear_mixing:
        from fhe_thesis.encryption.protocol import encrypt_inference_linear_mixing

        logits, timings = encrypt_inference_linear_mixing(
            backend,
            emb_np,
            weights,
            coeffs,
            max_seq_len=max_seq_len,
            n_jobs=n_jobs,
            kept_token_indices=kept_token_indices,
            bootstrap_plan=bootstrap_plan,
        )
    elif phase == "model":
        from fhe_thesis.encryption.protocol import encrypt_inference

        logits, timings = encrypt_inference(
            backend,
            emb_np,
            weights,
            coeffs,
            max_seq_len=max_seq_len,
            n_jobs=n_jobs,
            bootstrap_plan=bootstrap_plan,
        )
    else:
        from fhe_thesis.encryption.protocol import run_phase

        logits, timings = run_phase(
            phase,
            model,
            backend,
            emb_np,
            checkpoint_path=checkpoint,
            max_seq_len=max_seq_len,
            n_jobs=n_jobs,
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
    print(f"  linear_mixing={args.linear_mixing}  hybrid={args.hybrid}")
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
        "phase": args.phase,
        "linear_mixing": args.linear_mixing,
        "hybrid": args.hybrid,
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
    from fhe_thesis.encryption.coefficients import load_coefficients

    if args.hybrid:
        from fhe_thesis.encryption.protocol import load_hybrid_weights
        from fhe_thesis.encryption.hybrid_coefficients import load_coefficients_for_hybrid
        if args.checkpoint is None:
            from fhe_thesis.config import MULTI_MODEL_DIR
            args.checkpoint = str(
                MULTI_MODEL_DIR / args.task / args.model
                / "hybrid_progressive" / "best_model"
            )
        lm_layers = [int(x) for x in args.linear_mixing_layers.split(",") if x.strip()]
        qa_layers = [int(x) for x in args.quad_attention_layers.split(",") if x.strip()]
        print(f"  Hybrid checkpoint: {args.checkpoint}")
        print(f"  LinearMixing layers: {lm_layers}  Quad layers: {qa_layers}")
        weights = load_hybrid_weights(
            args.model,
            checkpoint_path=args.checkpoint,
            linear_mixing_layers=lm_layers,
            quad_attention_layers=qa_layers,
        )
        if args.reduced_degrees:
            coeffs = load_coefficients_for_hybrid(
                args.model, task=args.task,
                linear_mixing_layers=lm_layers,
                quad_attention_layers=qa_layers,
            )
        else:
            coeffs = load_coefficients(args.model, task=args.task)
            # Strip Softmax keys for non-LPAN layers (Quad/LM don't need them)
            for li in list(lm_layers) + list(qa_layers):
                if li in coeffs:
                    coeffs[li] = {k: v for k, v in coeffs[li].items() if k != "Softmax"}
    elif args.linear_mixing:
        from fhe_thesis.encryption.protocol import load_linear_mixing_weights
        lm_ckpt = args.linear_mixing_checkpoint
        if lm_ckpt is None:
            from fhe_thesis.config import MULTI_MODEL_DIR
            lm_ckpt = str(
                MULTI_MODEL_DIR / args.task / args.model
                / "linear_mixing_final" / "best_model"
            )
        print(f"  Linear mixing checkpoint: {lm_ckpt}")
        weights = load_linear_mixing_weights(
            args.model, checkpoint_path=lm_ckpt
        )
    else:
        from fhe_thesis.encryption.protocol import load_model_weights
        weights = load_model_weights(args.model, checkpoint_path=args.checkpoint)
        coeffs = load_coefficients(args.model, task=args.task)

    # Get embeddings (plaintext embedding layer runs outside FHE)
    print("Computing plaintext embeddings for all samples...")
    elim_active = args.word_elimination != "none" and (args.hybrid or args.linear_mixing)
    need_scores = args.word_elimination == "content_teacher" and elim_active
    embeddings, labels, masks, teacher_scores = _get_embeddings(
        args.model, samples, args.checkpoint, need_teacher_scores=need_scores,
    )
    if elim_active:
        print(f"  Word elimination: strategy={args.word_elimination} "
              f"keep_ratio={args.keep_ratio}")
        results["word_elimination"] = args.word_elimination
        results["keep_ratio"] = args.keep_ratio if args.word_elimination == "content_teacher" else None
    else:
        results["word_elimination"] = "none"

    # ── Bootstrap plan (Ext 1: region-adaptive scheduler) ─────────────
    bootstrap_plan = None
    if args.bootstrap_strategy != "none" and not args.no_bootstrap:
        from fhe_thesis.encryption.bootstrap_scheduler import (
            composition_to_kinds, schedule_bootstraps, schedule_uniform,
        )
        # Determine layer composition
        from fhe_thesis.config import MODEL_REGISTRY
        num_layers = MODEL_REGISTRY[args.model]["layers"]
        if args.hybrid:
            lm = [int(x) for x in args.linear_mixing_layers.split(",") if x.strip()]
            qa = [int(x) for x in args.quad_attention_layers.split(",") if x.strip()]
        elif args.linear_mixing:
            lm = list(range(num_layers)); qa = []
        else:
            lm = []; qa = []  # pure LPAN: all layers are 'L'
        kinds = composition_to_kinds(num_layers, lm, qa)
        if args.bootstrap_strategy == "adaptive":
            bootstrap_plan = schedule_bootstraps(kinds, args.bootstrap_budget)
        else:
            bootstrap_plan = schedule_uniform(num_layers, args.bootstrap_period,
                                              kinds, args.bootstrap_budget)
        print(f"\n[Bootstrap plan: {args.bootstrap_strategy}]")
        print(f"  layer_kinds       = {kinds}")
        print(f"  insertion_indices = {bootstrap_plan.insertion_indices}")
        print(f"  num_bootstraps    = {bootstrap_plan.num_bootstraps}")
        print(f"  total_depth       = {bootstrap_plan.total_depth}")
        print(f"  budget_per_window = {bootstrap_plan.budget_per_window}")
        results["bootstrap_plan"] = bootstrap_plan.to_dict()

    # ── FHE inference loop ────────────────────────────────────────────
    print(f"\nRunning FHE inference on {len(samples)} samples...")
    fhe_correct = 0
    all_latencies: List[float] = []
    all_timings: List[Dict] = []

    for idx, (emb, label) in enumerate(zip(embeddings, labels)):
        # Ext W: word elimination (FHE-pure, applied client-side before encryption)
        kept_indices = None
        emb_in = emb
        if elim_active:
            from fhe_thesis.encryption.elimination import (
                apply_elimination, elimination_savings,
            )
            t_scores = teacher_scores[idx] if teacher_scores is not None else None
            emb_in, kept_indices = apply_elimination(
                emb, masks[idx],
                strategy=args.word_elimination,
                teacher_scores=t_scores,
                keep_ratio=args.keep_ratio,
            )
            sav = elimination_savings(emb.shape[0], len(kept_indices))
            if idx == 0:
                print(f"  [elim] kept={sav['kept_tokens']}/{sav['orig_tokens']} "
                      f"(ratio={sav['keep_ratio']:.3f}, "
                      f"linear_x={sav['linear_speedup']:.2f}, "
                      f"quad_x={sav['quadratic_speedup']:.2f})")

        t_start = time.time()
        logits, timings = _run_fhe_sample(
            backend,
            weights,
            coeffs,
            emb_in,
            args.max_seq_len,
            args.n_jobs,
            args.phase,
            args.model,
            args.checkpoint,
            linear_mixing=args.linear_mixing,
            hybrid=args.hybrid,
            kept_token_indices=kept_indices,
            bootstrap_plan=bootstrap_plan,
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
