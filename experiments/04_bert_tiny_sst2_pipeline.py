#!/usr/bin/env python3
"""
Experiment 04: BERT-Tiny SST-2 Fine-Tuning — Baseline & Polynomial-Replaced
=============================================================================
1. Train baseline BERT-Tiny on SST-2
2. Replace activations with polynomial approximations
3. Evaluate zero-shot & fine-tuned

Outputs → results/training/, results/models/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fhe_thesis.config import (
    DEFAULT_MODEL_NAME, MODELS_DIR, TRAINING_DIR, ensure_dirs,
)
from fhe_thesis.models.replacement import replace_activations
from fhe_thesis.models.profiling import profile_model, compute_poly_coefficients
from fhe_thesis.training.trainer import (
    compute_metrics, detect_device, load_sst2_dataset, train_and_eval,
)


def main():
    parser = argparse.ArgumentParser(description="BERT-Tiny SST-2 fine-tuning")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 70)
    print("  BERT-Tiny SST-2 Fine-Tuning Pipeline")
    print("=" * 70)

    all_results = {}

    print("\n[0/4] Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    train_ds, eval_ds = load_sst2_dataset(tokenizer)
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Phase 1: Baseline
    print("\n[1/4] Training BASELINE model...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME, num_labels=2)
    baseline_res = train_and_eval(
        baseline_model, train_ds, eval_ds,
        str(MODELS_DIR / "baseline"), args.epochs, args.batch_size,
        args.lr, "Baseline (Standard BERT-Tiny)",
    )
    all_results["baseline"] = baseline_res

    if args.baseline_only:
        with open(TRAINING_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        return

    # Phase 2: Profile & fit polynomials
    print("\n[2/4] Profiling activations & fitting polynomials...")
    profile_data = profile_model(DEFAULT_MODEL_NAME, num_layers=2, num_samples=2000)
    poly_coeffs = compute_poly_coefficients(profile_data, num_layers=2)

    # Phase 3: Polynomial replacement
    print("\n[3/4] Polynomial replacement...")
    best_path = MODELS_DIR / "baseline" / "best_model"
    if best_path.exists():
        poly_model = AutoModelForSequenceClassification.from_pretrained(
            str(best_path), num_labels=2)
    else:
        poly_model = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_MODEL_NAME, num_labels=2)
        poly_model.load_state_dict(baseline_model.state_dict())

    replace_activations(poly_model, poly_coeffs)

    # Zero-shot eval
    device = detect_device()
    poly_model.to(device)
    zs_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "poly_zero_shot"),
        per_device_eval_batch_size=64, report_to="none",
        no_cuda=device.type == "cpu",
    )
    zs_trainer = Trainer(model=poly_model, args=zs_args, eval_dataset=eval_ds,
                         compute_metrics=compute_metrics)
    zs_eval = zs_trainer.evaluate()
    print(f"  Zero-shot accuracy: {zs_eval['eval_accuracy']:.4f}")
    all_results["poly_replaced_zero_shot"] = {
        "label": "Poly-Replaced (Zero-Shot)",
        "accuracy": zs_eval["eval_accuracy"],
        "eval_loss": zs_eval["eval_loss"], "epochs": 0,
    }

    # Fine-tune
    ft_res = train_and_eval(
        poly_model, train_ds, eval_ds,
        str(MODELS_DIR / "poly_finetuned"), args.epochs, args.batch_size,
        args.lr * 0.5, "Poly-Replaced (Fine-Tuned)",
    )
    all_results["poly_replaced_finetuned"] = ft_res

    # Phase 4: Summary
    print("\n[4/4] Generating summary...")
    with open(TRAINING_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    for key, res in all_results.items():
        print(f"  {res['label']:40s}: {res['accuracy']:.4f}")
    base_acc = all_results["baseline"]["accuracy"]
    ft_acc = all_results["poly_replaced_finetuned"]["accuracy"]
    print(f"\n  Fine-tuned accuracy drop: {(base_acc - ft_acc)*100:+.2f}%")
    print(f"\n  Results saved to: {TRAINING_DIR}/")


if __name__ == "__main__":
    main()
