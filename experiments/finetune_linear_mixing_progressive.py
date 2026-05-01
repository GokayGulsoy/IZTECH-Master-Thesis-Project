#!/usr/bin/env python3
"""Progressive layer-by-layer linear mixing replacement with knowledge distillation.

The key insight from LPAN's success (91.40% on SST-2) is PROGRESSIVE replacement:
replace ONE layer's attention at a time, fine-tune, then move to the next.

This preserves accuracy because:
  1. Identity initialization: each new mixing layer starts as pass-through
  2. Local adaptation: FFN in the current layer adapts to new mixing output
  3. Co-adaptation: previously replaced mixing layers continue training
  4. Small perturbation: only 1/12 of the model changes per step

Strategy:
  Stage 4a — Progressive replacement (L0 → L11):
    For each layer i:
      - Replace layer i's attention with LinearMixingAttention
      - Unfreeze: all replaced layers' mixing params + layer i's FFN + classifier
      - Train with Hidden-State KD (AttentionDistillationTrainer, gamma=5.0)
      - Depth-adaptive epochs and LLRD LR scaling

  Stage 4b — Global fine-tune:
    - Unfreeze all mixing + all FFN + classifier
    - Train 2 epochs with lower LR, pure CE
    - Lets all layers co-adapt globally

Usage:
    python experiments/finetune_linear_mixing_progressive.py \\
        --model base --task sst2 --max-seq-len 64 \\
        --epochs-per-layer 3 --lr 1e-4 --batch-size 32

Resume from layer 6:
    python experiments/finetune_linear_mixing_progressive.py \\
        --model base --task sst2 --start-layer 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
    LinearMixingAttention,
    freeze_for_progressive_mixing,
    replace_attention_with_linear_mixing,
)
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.training.trainer import (
    AttentionDistillationTrainer,
    NaNSafeTrainer,
    attn_distill_and_eval,
    compute_metrics,
)


TASK_CONFIG = {
    "sst2": {"text_a": "sentence", "text_b": None},
    "mrpc": {"text_a": "sentence1", "text_b": "sentence2"},
    "qnli": {"text_a": "question", "text_b": "sentence"},
}


def _unfreeze_all_ffns_and_mixing(model, replaced_layers):
    """Unfreeze ALL FFNs + all mixing params + classifier for deep-layer training.

    When many layers are replaced, earlier FFNs need to co-adapt to the changed
    representation pipeline. This gives the model maximum flexibility.
    """
    replaced_set = set(replaced_layers)
    trainable = 0

    for name, param in model.named_parameters():
        should_train = False

        if name.startswith("classifier.") or name.startswith("bert.pooler."):
            should_train = True
        elif name.startswith("bert.encoder.layer."):
            parts = name.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])

            # All mixing params in replaced layers
            if li in replaced_set and rest.startswith("attention."):
                should_train = True

            # ALL layers' FFN params (intermediate + output)
            if rest.startswith("intermediate.") or rest.startswith("output."):
                should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()

    return trainable


def _unfreeze_replaced_ffns_and_mixing(model, replaced_layers):
    """Unfreeze FFNs of replaced layers + all mixing params + classifier.

    LPAN-inspired co-adaptation: when layer i is replaced, ALL previously
    replaced layers' FFNs + mixing params stay trainable so they can compensate
    for the cascading perturbation. Non-replaced layers stay frozen.

    This is the middle ground between:
    - freeze_for_progressive_mixing (only current FFN — too restrictive)
    - _unfreeze_all_ffns_and_mixing (all 12 FFNs — too aggressive early on)
    """
    replaced_set = set(replaced_layers)
    trainable = 0

    for name, param in model.named_parameters():
        should_train = False

        if name.startswith("classifier.") or name.startswith("bert.pooler."):
            should_train = True
        elif name.startswith("bert.encoder.layer."):
            parts = name.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])

            if li in replaced_set:
                # Replaced layers: unfreeze mixing + FFN
                if rest.startswith("attention."):
                    should_train = True
                if rest.startswith("intermediate.") or rest.startswith("output."):
                    should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()

    return trainable


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


def _load_checkpoint_state(ckpt_dir: Path) -> dict:
    """Load model state dict from safetensors or pytorch_model.bin."""
    sf_path = ckpt_dir / "model.safetensors"
    bin_path = ckpt_dir / "pytorch_model.bin"
    if sf_path.exists():
        from safetensors.torch import load_file
        return load_file(str(sf_path))
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Progressive linear mixing replacement with KD"
    )
    parser.add_argument("--model", default="base",
                        choices=["tiny", "mini", "small", "base"])
    parser.add_argument("--task", default="sst2",
                        choices=["sst2", "mrpc", "qnli"])
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--epochs-per-layer", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override LPAN checkpoint path")
    parser.add_argument("--start-layer", type=int, default=0,
                        help="Resume from this layer (previous layers loaded from disk)")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Hidden-state MSE weight (0 = pure CE)")
    parser.add_argument("--gamma-decay", action="store_true",
                        help="Decay gamma for deep layers: full for L0-L3, "
                             "0.5x for L4-L7, 0.25x for L8-L11")
    parser.add_argument("--global-gamma", type=float, default=2.0,
                        help="Hidden-state MSE weight for global fine-tune (0 = pure CE)")
    parser.add_argument("--lr-schedule", type=str, default="linear",
                        choices=["linear", "cosine", "constant_with_warmup"],
                        help="LR scheduler type (default: linear)")
    parser.add_argument("--unfreeze-all-ffn-from", type=int, default=-1,
                        help="Unfreeze ALL FFNs starting from this layer index "
                             "(-1 = never, only current FFN). Recommended: 6")
    parser.add_argument("--skip-global-finetune", action="store_true",
                        help="Skip the final global fine-tune stage")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = MODEL_REGISTRY[args.model]
    model_name = cfg["name"]
    num_layers = cfg["layers"]

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
        MULTI_MODEL_DIR / args.task / args.model / "linear_mixing_progressive"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Progressive Linear Mixing Replacement")
    print(f"{'='*60}")
    print(f"  Model: {cfg['short']}  Task: {args.task}")
    print(f"  LPAN checkpoint: {ckpt_path}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  epochs/layer: {args.epochs_per_layer}  final: {args.final_epochs}")
    print(f"  lr: {args.lr}  gamma(HidMSE): {args.gamma}  schedule: {args.lr_schedule}")
    print(f"  gamma_decay: {args.gamma_decay}  unfreeze_all_ffn_from: {args.unfreeze_all_ffn_from}")
    print(f"  Device: {device}")
    if args.start_layer > 0:
        print(f"  Resuming from layer {args.start_layer}")
    print(f"{'='*60}")

    # ── 1. Load LPAN model (student) ──────────────────────────────────────
    print("\n[1/4] Loading LPAN model (student)...")
    model = load_lpan_model(
        args.model, ckpt_path, device="cpu",
        profile_samples=200, degree=8,
    )

    # ── 2. Load data & evaluate baseline ──────────────────────────────────
    print("\n[2/4] Loading data & evaluating LPAN baseline...")
    train_ds, eval_ds = load_data(model_name, args.task, args.max_seq_len)
    model.to(device)

    baseline_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "_baseline_eval"),
            per_device_eval_batch_size=64,
            report_to="none", disable_tqdm=True,
        ),
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    baseline_metrics = baseline_trainer.evaluate()
    lpan_acc = baseline_metrics["eval_accuracy"]
    print(f"  LPAN baseline: {lpan_acc:.4f} ({lpan_acc:.2%})")

    # Load teacher (static LPAN — frozen throughout)
    print("  Loading teacher (LPAN model, frozen)...")
    teacher = load_lpan_model(
        args.model, ckpt_path, device=device,
        profile_samples=200, degree=8,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {n_teacher:,} params (frozen)")

    # ── 3. Progressive layer-by-layer replacement ─────────────────────────
    print(f"\n[3/4] Progressive replacement (L{args.start_layer} → L{num_layers-1})...")
    model.cpu()

    # Resume: replace + load weights for previously completed layers
    replaced_so_far: list[int] = []
    if args.start_layer > 0:
        for prev_li in range(args.start_layer):
            replace_attention_with_linear_mixing(
                model, args.max_seq_len, layer_indices=[prev_li],
            )
            replaced_so_far.append(prev_li)

        # Load full state dict from the last completed layer's checkpoint
        last_ckpt = output_dir / f"layer_{args.start_layer - 1}" / "best_model"
        if last_ckpt.exists():
            state = _load_checkpoint_state(last_ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  Restored from L{args.start_layer-1} checkpoint "
                  f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
        else:
            print(f"  WARNING: No checkpoint at {last_ckpt}, "
                  f"starting from LPAN weights for layers 0-{args.start_layer-1}")

    # ── Layer-by-layer loop ───────────────────────────────────────────────
    layer_results: list[dict] = []

    for li in range(args.start_layer, num_layers):
        # Flat epochs per layer — co-adaptation handles error accumulation
        layer_epochs = args.epochs_per_layer

        # LLRD-inspired LR scaling: 1.0× at L0 → 1.5× at L11
        lr_scale = 1.0 + 0.5 * (li / max(1, num_layers - 1))
        layer_lr = args.lr * lr_scale

        # Gamma decay: reduce hidden-state MSE weight for deep layers
        # Deep layers' teacher hidden states diverge too much — prioritize CE
        if args.gamma_decay:
            third = num_layers // 3
            if li < third:        # L0-L3: full gamma
                layer_gamma = args.gamma
            elif li < 2 * third:  # L4-L7: half gamma
                layer_gamma = args.gamma * 0.5
            else:                 # L8-L11: quarter gamma
                layer_gamma = args.gamma * 0.25
        else:
            layer_gamma = args.gamma

        print(f"\n  {'─'*50}")
        print(f"  Layer {li}/{num_layers-1}  "
              f"epochs={layer_epochs}  lr={layer_lr:.2e}  gamma={layer_gamma:.2f}")
        print(f"  {'─'*50}")

        # Replace attention in this layer only
        model.cpu()
        replace_attention_with_linear_mixing(
            model, args.max_seq_len, layer_indices=[li],
        )
        replaced_so_far.append(li)

        # Freeze strategy (LPAN-inspired co-adaptation):
        # - Default: unfreeze mixing + FFN for ALL replaced layers (graduated)
        # - With --unfreeze-all-ffn-from N: unfreeze ALL 12 FFNs from layer N+
        if args.unfreeze_all_ffn_from >= 0 and li >= args.unfreeze_all_ffn_from:
            # Aggressive: unfreeze ALL FFNs for global co-adaptation
            trainable = _unfreeze_all_ffns_and_mixing(model, replaced_so_far)
        else:
            # LPAN-style: unfreeze FFN + mixing for all replaced layers
            trainable = _unfreeze_replaced_ffns_and_mixing(model, replaced_so_far)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total_params:,} "
              f"({100*trainable/total_params:.1f}%)")

        model.to(device)

        # Fine-tune with hidden-state KD
        sub_output = str(output_dir / f"layer_{li}")
        result = attn_distill_and_eval(
            student_model=model,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=sub_output,
            epochs=layer_epochs,
            batch_size=args.batch_size,
            lr=layer_lr,
            label=f"L{li} mixing",
            use_fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            alpha=1.0,
            beta=0.0,     # No attention KL — mixing doesn't produce attention
            gamma=layer_gamma,  # Hidden-state MSE (decayed for deep layers)
            seed=args.seed,
            lr_scheduler_type=args.lr_schedule,
        )

        layer_acc = result["accuracy"]
        drop = (lpan_acc - layer_acc) * 100
        layer_results.append({
            "layer": li,
            "accuracy": layer_acc,
            "drop_from_lpan": drop,
            "epochs": layer_epochs,
            "lr": layer_lr,
        })
        print(f"  → L{li} accuracy: {layer_acc:.4f}  (drop: {drop:+.2f}%)")

        # Clean up training checkpoints (keep best_model only)
        training_dir = Path(sub_output)
        for item in training_dir.iterdir():
            if item.name != "best_model" and item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    # Save progressive-final checkpoint
    prog_path = output_dir / "progressive_final" / "best_model"
    prog_path.mkdir(parents=True, exist_ok=True)
    for p in model.parameters():
        p.data = p.data.contiguous()
    model.save_pretrained(str(prog_path))
    prog_acc = layer_results[-1]["accuracy"] if layer_results else lpan_acc
    print(f"\n  Progressive checkpoint saved ({prog_acc:.4f})")

    # ── 4. Global fine-tune ───────────────────────────────────────────────
    final_acc = prog_acc

    if not args.skip_global_finetune:
        global_lr = args.lr / 3
        print(f"\n[4/4] Global fine-tune "
              f"({args.final_epochs} epochs, lr={global_lr:.2e})...")

        # Unfreeze entire encoder + classifier for global co-adaptation
        trainable = 0
        for name, param in model.named_parameters():
            should_train = (
                name.startswith("bert.encoder.")
                or name.startswith("classifier.")
                or name.startswith("bert.pooler.")
            )
            param.requires_grad = should_train
            if should_train:
                trainable += param.numel()
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")

        global_output = str(output_dir / "global_finetune")
        global_result = attn_distill_and_eval(
            student_model=model,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=global_output,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            lr=global_lr,
            label="Global fine-tune",
            use_fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            alpha=1.0,
            beta=0.0,
            gamma=args.global_gamma,
            seed=args.seed,
            lr_scheduler_type=args.lr_schedule,
        )
        final_acc = global_result["accuracy"]
        print(f"  → Global fine-tune accuracy: {final_acc:.4f}")

        # Clean up
        global_dir = Path(global_output)
        for item in global_dir.iterdir():
            if item.name != "best_model" and item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    # ── Save final model + results ────────────────────────────────────────
    final_path = output_dir / "best_model"
    final_path.mkdir(parents=True, exist_ok=True)
    for p in model.parameters():
        p.data = p.data.contiguous()
    model.save_pretrained(str(final_path))

    results = {
        "model": args.model,
        "task": args.task,
        "max_seq_len": args.max_seq_len,
        "lpan_accuracy": lpan_acc,
        "per_layer_results": layer_results,
        "progressive_accuracy": prog_acc,
        "final_accuracy": final_acc,
        "accuracy_drop_from_lpan": (lpan_acc - final_acc) * 100,
        "epochs_per_layer": args.epochs_per_layer,
        "final_epochs": args.final_epochs,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    drop = (lpan_acc - final_acc) * 100
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  LPAN baseline:      {lpan_acc:.4f}")
    for r in layer_results:
        print(f"  After L{r['layer']:2d}:          "
              f"{r['accuracy']:.4f}  ({r['drop_from_lpan']:+.2f}%)")
    print(f"  Progressive final:  {prog_acc:.4f}")
    if not args.skip_global_finetune:
        print(f"  Global fine-tune:   {final_acc:.4f}")
    print(f"  Drop from LPAN:     {drop:+.2f}%")
    print(f"  Saved to: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
