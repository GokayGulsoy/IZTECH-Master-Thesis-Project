#!/usr/bin/env python3
"""Fine-tune linear mixing layers on an LPAN-trained BERT checkpoint.

Loads a fully polynomial LPAN model (Stage 3 final), replaces all
attention blocks with LinearMixingAttention, freezes everything except
the mixing layers + classifier, and fine-tunes for a few epochs.

This is Stage 4 of the LPAN pipeline: attention elimination for
FHE-friendly inference.

Usage (local GPU — RTX 5070 Ti or any GPU with >=4GB VRAM):
    python experiments/finetune_linear_mixing.py \\
        --model base --task sst2 --max-seq-len 64 \\
        --epochs 5 --lr 2e-4 --batch-size 32

With Knowledge Distillation (recommended — uses LPAN model as teacher):
    python experiments/finetune_linear_mixing.py \\
        --model base --task sst2 --max-seq-len 64 \\
        --epochs 5 --lr 2e-4 --batch-size 32 \\
        --kd --kd-alpha 0.5 --kd-temperature 4.0

The output checkpoint is saved to:
    results/multi_model/<task>/<model>/linear_mixing_final/best_model/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR
from fhe_thesis.models.linear_mixing import (
    freeze_for_mixing_finetune,
    replace_attention_with_linear_mixing,
)
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.training.trainer import (
    DistillationTrainer,
    NaNSafeTrainer,
    compute_metrics,
)


TASK_CONFIG = {
    "sst2": {"text_a": "sentence", "text_b": None},
    "mrpc": {"text_a": "sentence1", "text_b": "sentence2"},
    "qnli": {"text_a": "question", "text_b": "sentence"},
}


def load_data(model_name: str, task: str, max_seq_len: int):
    """Load and tokenize a GLUE task dataset."""
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", task)
    col_a = TASK_CONFIG[task]["text_a"]
    col_b = TASK_CONFIG[task]["text_b"]

    def tokenize_fn(ex):
        if col_b is None:
            return tokenizer(
                ex[col_a], truncation=True, padding="max_length",
                max_length=max_seq_len,
            )
        return tokenizer(
            ex[col_a], ex[col_b], truncation=True, padding="max_length",
            max_length=max_seq_len,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized["train"], tokenized["validation"]


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune linear mixing on LPAN checkpoint"
    )
    parser.add_argument("--model", default="base",
                        choices=["tiny", "mini", "small", "base"])
    parser.add_argument("--task", default="sst2",
                        choices=["sst2", "mrpc", "qnli"])
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override LPAN checkpoint path")
    # Knowledge distillation options
    parser.add_argument("--kd", action="store_true",
                        help="Use LPAN model as teacher for knowledge distillation")
    parser.add_argument("--kd-alpha", type=float, default=0.5,
                        help="KD loss weight: alpha*CE + (1-alpha)*KL (default: 0.5)")
    parser.add_argument("--kd-temperature", type=float, default=4.0,
                        help="KD softmax temperature (default: 4.0)")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = MODEL_REGISTRY[args.model]
    model_name = cfg["name"]

    # Resolve LPAN checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = (
            MULTI_MODEL_DIR / args.task / args.model
            / "staged_lpan_final" / "best_model"
        )
    if not ckpt_path.exists():
        print(f"ERROR: LPAN checkpoint not found at {ckpt_path}")
        sys.exit(1)

    output_dir = (
        MULTI_MODEL_DIR / args.task / args.model / "linear_mixing_final"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Linear Mixing Fine-Tuning ===")
    print(f"  Model: {cfg['short']}  Task: {args.task}")
    print(f"  LPAN checkpoint: {ckpt_path}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  KD: {args.kd}" + (f"  alpha={args.kd_alpha}  T={args.kd_temperature}" if args.kd else ""))
    print(f"  Device: {device}")

    # 1. Load LPAN model with all polynomial activations
    print("\n[1/5] Loading LPAN model...")
    model = load_lpan_model(
        args.model, ckpt_path, device="cpu",
        profile_samples=200, degree=8,
    )

    # 2. Evaluate LPAN baseline accuracy before modification
    print("\n[2/5] Loading data & evaluating LPAN baseline...")
    train_ds, eval_ds = load_data(model_name, args.task, args.max_seq_len)
    model.to(device)

    baseline_args = TrainingArguments(
        output_dir=str(output_dir / "_baseline_eval"),
        per_device_eval_batch_size=64,
        report_to="none", disable_tqdm=True,
    )
    baseline_trainer = Trainer(
        model=model, args=baseline_args, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    baseline_metrics = baseline_trainer.evaluate()
    lpan_acc = baseline_metrics["eval_accuracy"]
    print(f"  LPAN baseline accuracy: {lpan_acc:.4f} ({lpan_acc:.2%})")

    # 2b. If KD, load a second copy of LPAN as the frozen teacher
    teacher = None
    if args.kd:
        print("  Loading LPAN teacher for knowledge distillation...")
        teacher = load_lpan_model(
            args.model, ckpt_path, device=device,
            profile_samples=200, degree=8,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Teacher loaded ({sum(p.numel() for p in teacher.parameters()):,} params, frozen)")

    # 3. Replace attention with linear mixing (student only)
    print("\n[3/5] Replacing attention with linear mixing...")
    model.cpu()
    replace_attention_with_linear_mixing(model, max_seq_len=args.max_seq_len)
    trainable = freeze_for_mixing_finetune(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} "
          f"({100*trainable/total_params:.1f}%)")

    # Zero-shot eval after replacement (should be close to LPAN baseline
    # because mixing layers are initialized to identity)
    model.to(device)
    zs_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "_zs_eval"),
            per_device_eval_batch_size=64,
            report_to="none", disable_tqdm=True,
        ),
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    zs_metrics = zs_trainer.evaluate()
    zs_acc = zs_metrics["eval_accuracy"]
    print(f"  After replacement (zero-shot): {zs_acc:.4f} ({zs_acc:.2%})")

    # 4. Fine-tune mixing layers
    print(f"\n[4/5] Fine-tuning for {args.epochs} epochs "
          f"(lr={args.lr}, bs={args.batch_size})...")
    training_args = TrainingArguments(
        output_dir=str(output_dir / "training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=args.seed,
    )

    if args.kd and teacher is not None:
        print(f"  Using DistillationTrainer (alpha={args.kd_alpha}, T={args.kd_temperature})")
        trainer = DistillationTrainer(
            teacher_model=teacher,
            alpha=args.kd_alpha,
            temperature=args.kd_temperature,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = NaNSafeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
    trainer.train()
    final_metrics = trainer.evaluate()
    final_acc = final_metrics["eval_accuracy"]
    print(f"  Final accuracy: {final_acc:.4f} ({final_acc:.2%})")

    # 5. Save
    print(f"\n[5/5] Saving checkpoint...")
    best_path = str(output_dir / "best_model")
    os.makedirs(best_path, exist_ok=True)
    for p in model.parameters():
        p.data = p.data.contiguous()
    model.save_pretrained(best_path)

    # Save results summary
    results = {
        "model": args.model,
        "task": args.task,
        "max_seq_len": args.max_seq_len,
        "kd": args.kd,
        "kd_alpha": args.kd_alpha if args.kd else None,
        "kd_temperature": args.kd_temperature if args.kd else None,
        "lpan_accuracy": lpan_acc,
        "zero_shot_accuracy": zs_acc,
        "final_accuracy": final_acc,
        "accuracy_drop_from_lpan": (lpan_acc - final_acc) * 100,
        "trainable_params": trainable,
        "total_params": total_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    drop = (lpan_acc - final_acc) * 100
    print(f"\n=== Results ===")
    print(f"  LPAN baseline:   {lpan_acc:.4f}")
    print(f"  Zero-shot:       {zs_acc:.4f}")
    print(f"  After fine-tune: {final_acc:.4f}")
    print(f"  Drop from LPAN:  {drop:+.2f}%")
    print(f"  Saved to: {best_path}")


if __name__ == "__main__":
    main()
