#!/usr/bin/env python3
"""HyPER-LPAN: Progressive hybrid attention replacement with knowledge distillation.

This is the main "new SOTA" training pipeline combining:
  - LPAN polynomial-softmax attention in deep layers (preserved)
  - QuadAttention (2Quad, MPCFormer-style) in mid layers
  - LinearMixingAttention (your work) in early layers

Strategy (3 stages):
  Stage 1 — Mid-layer 2Quad replacement (L4-L7 by default):
    Replace mid-layer attention with 2Quad. KD from LPAN teacher.
    Co-adapt: unfreeze attention + FFNs of all replaced layers.
    Expected accuracy: ~91% (close to LPAN baseline).

  Stage 2 — Early-layer linear mixing replacement (L0-L3 by default):
    Replace early-layer attention with multi-head linear mixing.
    Continue co-adapting all replaced layers.
    Expected accuracy: ~90.5-91%.

  Stage 3 — Joint global fine-tune:
    Unfreeze entire encoder + classifier. Pure CE + small HidMSE.
    Expected accuracy: ~91% (recovers any residual gap).

Each stage uses progressive layer-by-layer replacement to minimize the
perturbation to the model at each step.

Usage:
    python experiments/finetune_hybrid_progressive.py \\
        --model base --task sst2 --max-seq-len 64 \\
        --linear-mixing-layers 0,1,2,3 \\
        --quad-attention-layers 4,5,6,7 \\
        --epochs-per-layer 4 --lr 8e-5 --batch-size 32 \\
        --gamma 4.0 --gamma-decay \\
        --global-gamma 2.0 --final-epochs 4 \\
        --lr-schedule constant_with_warmup
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR
from fhe_thesis.models.linear_mixing import (
    MultiHeadLinearMixingAttention,
    replace_attention_with_linear_mixing,
)
from fhe_thesis.models.quad_attention import (
    QuadAttention,
    replace_attention_with_quad,
)
from fhe_thesis.models.hybrid_attention import (
    apply_hybrid_attention,
    freeze_for_progressive_hybrid,
    freeze_for_global_finetune,
    summarize_attention_types,
)
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.training.trainer import (
    attn_distill_and_eval,
    compute_metrics,
)


TASK_CONFIG = {
    "sst2": {"text_a": "sentence", "text_b": None},
    "mrpc": {"text_a": "sentence1", "text_b": "sentence2"},
    "qnli": {"text_a": "question", "text_b": "sentence"},
}


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers like '0,1,2,3' -> [0,1,2,3]."""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_data(model_name: str, task: str, max_seq_len: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

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


def compute_layer_gamma(li: int, num_layers: int, base_gamma: float, decay: bool) -> float:
    """Per-layer gamma: full for early thirds, half for middle, quarter for deep."""
    if not decay:
        return base_gamma
    third = num_layers // 3
    if li < third:
        return base_gamma
    elif li < 2 * third:
        return base_gamma * 0.5
    else:
        return base_gamma * 0.25


def replace_layer(
    model,
    li: int,
    layer_type: str,
    max_seq_len: int,
    num_heads: int,
) -> None:
    """Replace a single layer with the specified attention type."""
    if layer_type == "linear_mixing":
        replace_attention_with_linear_mixing(
            model, max_seq_len=max_seq_len, layer_indices=[li], num_heads=num_heads
        )
    elif layer_type == "quad":
        replace_attention_with_quad(
            model, layer_indices=[li], num_heads=num_heads, init_from_original=True,
        )
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


def _load_checkpoint_into(model, ckpt_path: Path, device: str) -> None:
    """Load weights from a checkpoint directory directly into model in-place.

    The model must already have the correct architecture (i.e. all attention
    replacements applied) before calling this, so state_dict keys match.
    """
    state_path = ckpt_path / "model.safetensors"
    if state_path.exists():
        import safetensors.torch
        state_dict = safetensors.torch.load_file(str(state_path), device="cpu")
    else:
        state_path = ckpt_path / "pytorch_model.bin"
        state_dict = torch.load(str(state_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)


def progressive_stage(
    model,
    teacher,
    train_ds,
    eval_ds,
    args,
    stage_name: str,
    layer_order: list[tuple[int, str]],
    replaced_so_far: list[int],
    output_dir: Path,
    num_layers: int,
    lpan_acc: float,
    device: str,
    skip_layers: set[int] | None = None,
) -> tuple[list[dict], float]:
    """Run one progressive stage: replace each layer in layer_order, KD-finetune.

    Parameters
    ----------
    layer_order : list of (layer_index, layer_type) tuples
        e.g. [(4, 'quad'), (5, 'quad'), (6, 'quad'), (7, 'quad')]
    replaced_so_far : list[int]
        Updated in-place with newly replaced layers.
    skip_layers : set of layer indices to skip (already trained).
        The LAST skipped layer's best_model checkpoint will be loaded.

    Returns: (per_layer_results, last_accuracy)
    """
    layer_results = []
    last_acc = lpan_acc
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    skip_layers = skip_layers or set()

    # --- Resume: apply all skipped layers' replacements and load last checkpoint ---
    skipped_in_order = [(li, lt) for li, lt in layer_order if li in skip_layers]
    if skipped_in_order:
        print(f"  Resuming: skipping already-trained layers {[li for li,_ in skipped_in_order]}")
        model.cpu()
        for li, layer_type in skipped_in_order:
            replace_layer(model, li, layer_type, args.max_seq_len, num_heads=None)
            replaced_so_far.append(li)
        # Load weights from the last completed checkpoint
        last_li, _ = skipped_in_order[-1]
        last_ckpt = stage_dir / f"layer_{last_li}" / "best_model"
        if last_ckpt.exists():
            print(f"  Loading checkpoint from {last_ckpt}")
            _load_checkpoint_into(model, last_ckpt, device)
        else:
            print(f"  WARNING: checkpoint {last_ckpt} not found, starting from current weights")
            model.to(device)

    for li, layer_type in layer_order:
        if li in skip_layers:
            continue
        layer_epochs = args.epochs_per_layer
        # LLRD: 1.0× at L0 → 1.5× at L11
        lr_scale = 1.0 + 0.5 * (li / max(1, num_layers - 1))
        layer_lr = args.lr * lr_scale
        layer_gamma = compute_layer_gamma(li, num_layers, args.gamma, args.gamma_decay)

        print(f"\n  {'─'*60}")
        print(f"  [{stage_name}] Layer {li} ({layer_type})  "
              f"epochs={layer_epochs}  lr={layer_lr:.2e}  gamma={layer_gamma:.2f}")
        print(f"  {'─'*60}")

        # Replace this layer
        model.cpu()
        replace_layer(model, li, layer_type, args.max_seq_len, num_heads=None)
        replaced_so_far.append(li)

        # Co-adapt: unfreeze attention + FFNs of all replaced layers
        trainable = freeze_for_progressive_hybrid(
            model, replaced_layers=replaced_so_far, unfreeze_all_replaced_ffns=True,
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        model.to(device)

        # Fine-tune
        sub_output = str(stage_dir / f"layer_{li}")
        result = attn_distill_and_eval(
            student_model=model,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=sub_output,
            epochs=layer_epochs,
            batch_size=args.batch_size,
            lr=layer_lr,
            label=f"L{li} {layer_type}",
            use_fp16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            alpha=1.0,
            beta=0.0,
            gamma=layer_gamma,
            seed=args.seed,
            lr_scheduler_type=args.lr_schedule,
        )

        last_acc = result["accuracy"]
        drop = (lpan_acc - last_acc) * 100
        layer_results.append({
            "stage": stage_name,
            "layer": li,
            "layer_type": layer_type,
            "accuracy": last_acc,
            "drop_from_lpan": drop,
            "epochs": layer_epochs,
            "lr": layer_lr,
            "gamma": layer_gamma,
        })
        print(f"  → L{li} ({layer_type}) accuracy: {last_acc:.4f}  (drop: {drop:+.2f}%)")

        # Cleanup intermediate trainer checkpoints
        training_dir = Path(sub_output)
        for item in training_dir.iterdir():
            if item.name != "best_model" and item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    return layer_results, last_acc


def main():
    parser = argparse.ArgumentParser(
        description="HyPER-LPAN progressive hybrid attention replacement"
    )
    parser.add_argument("--model", default="base",
                        choices=["tiny", "mini", "small", "base"])
    parser.add_argument("--task", default="sst2",
                        choices=["sst2", "mrpc", "qnli"])
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--epochs-per-layer", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override LPAN checkpoint path")

    # Hybrid composition
    parser.add_argument("--linear-mixing-layers", type=str, default="0,1,2,3",
                        help="Comma-separated layer indices for linear mixing")
    parser.add_argument("--quad-attention-layers", type=str, default="4,5,6,7",
                        help="Comma-separated layer indices for 2Quad attention")

    # Stage ordering
    parser.add_argument("--stage-order", type=str, default="quad_first",
                        choices=["quad_first", "linear_first"],
                        help="quad_first: replace quad layers, then linear mixing")

    # Training schedule
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Hidden-state MSE weight (0 = pure CE)")
    parser.add_argument("--gamma-decay", action="store_true",
                        help="Decay gamma for deep layers")
    parser.add_argument("--global-gamma", type=float, default=2.0,
                        help="HidMSE weight for global fine-tune")
    parser.add_argument("--lr-schedule", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "constant_with_warmup"])
    parser.add_argument("--skip-global-finetune", action="store_true")
    parser.add_argument("--skip-stage-1", action="store_true",
                        help="Skip Stage 1 (quad replacement)")
    parser.add_argument("--skip-stage-2", action="store_true",
                        help="Skip Stage 2 (linear mixing replacement)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to a saved layer checkpoint to resume from. "
                             "All layers whose checkpoints exist under the stage dir "
                             "will be skipped and weights loaded from the last one.")
    parser.add_argument("--resume-skip-layers", type=str, default="",
                        help="Comma-separated layer indices to skip (already trained). "
                             "The last skipped layer's checkpoint must exist and will be loaded.")

    args = parser.parse_args()

    set_seed(args.seed)
    cfg = MODEL_REGISTRY[args.model]
    model_name = cfg["name"]
    num_layers = cfg["layers"]

    linear_mixing_layers = parse_int_list(args.linear_mixing_layers)
    quad_attention_layers = parse_int_list(args.quad_attention_layers)
    overlap = set(linear_mixing_layers) & set(quad_attention_layers)
    if overlap:
        print(f"ERROR: Layers {overlap} cannot be both linear-mixing and quad")
        sys.exit(1)
    lpan_layers = sorted(
        set(range(num_layers)) - set(linear_mixing_layers) - set(quad_attention_layers)
    )

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
        MULTI_MODEL_DIR / args.task / args.model / "hybrid_progressive"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  HyPER-LPAN: Progressive Hybrid Attention Replacement")
    print(f"{'='*70}")
    print(f"  Model: {cfg['short']}  Task: {args.task}")
    print(f"  LPAN checkpoint: {ckpt_path}")
    print(f"  Layer composition:")
    print(f"    Linear mixing: {linear_mixing_layers}")
    print(f"    2Quad attn:    {quad_attention_layers}")
    print(f"    LPAN (kept):   {lpan_layers}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  epochs/layer: {args.epochs_per_layer}  final: {args.final_epochs}")
    print(f"  lr: {args.lr}  gamma: {args.gamma}  decay: {args.gamma_decay}")
    print(f"  schedule: {args.lr_schedule}  global_gamma: {args.global_gamma}")
    print(f"  stage_order: {args.stage_order}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # ── 1. Load LPAN model (student) ──────────────────────────────────────
    print("\n[1/5] Loading LPAN model (student)...")
    model = load_lpan_model(
        args.model, ckpt_path, device="cpu",
        profile_samples=200, degree=8,
    )

    # ── 2. Load data & evaluate baseline ──────────────────────────────────
    print("\n[2/5] Loading data & evaluating LPAN baseline...")
    train_ds, eval_ds = load_data(model_name, args.task, args.max_seq_len)
    model.to(device)

    baseline_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "_baseline_eval"),
            per_device_eval_batch_size=args.batch_size * 2,
            report_to="none", disable_tqdm=True,
        ),
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    baseline = baseline_trainer.evaluate()
    lpan_acc = baseline["eval_accuracy"]
    print(f"  LPAN baseline: {lpan_acc:.4f} ({lpan_acc:.2%})")

    # Cleanup baseline tmp dir
    shutil.rmtree(output_dir / "_baseline_eval", ignore_errors=True)

    # ── 3. Load teacher (LPAN, frozen) ────────────────────────────────────
    print("\n[3/5] Loading teacher (LPAN model, frozen)...")
    teacher = load_lpan_model(
        args.model, ckpt_path, device=device,
        profile_samples=200, degree=8,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {n_teacher:,} params (frozen)")

    # ── 4. Progressive multi-stage replacement ────────────────────────────
    all_layer_results = []
    replaced_so_far: list[int] = []
    last_acc = lpan_acc

    # Parse resume-skip-layers
    resume_skip = set(parse_int_list(args.resume_skip_layers)) if args.resume_skip_layers else set()

    if args.stage_order == "quad_first":
        stage_specs = [
            ("stage1_quad", [(li, "quad") for li in sorted(quad_attention_layers)]),
            ("stage2_linear", [(li, "linear_mixing") for li in sorted(linear_mixing_layers)]),
        ]
    else:
        stage_specs = [
            ("stage1_linear", [(li, "linear_mixing") for li in sorted(linear_mixing_layers)]),
            ("stage2_quad", [(li, "quad") for li in sorted(quad_attention_layers)]),
        ]

    for stage_idx, (stage_name, layer_order) in enumerate(stage_specs, start=1):
        # Skip flags
        if stage_idx == 1 and args.skip_stage_1:
            print(f"\n[4/5] SKIPPING {stage_name}")
            continue
        if stage_idx == 2 and args.skip_stage_2:
            print(f"\n[4/5] SKIPPING {stage_name}")
            continue
        if not layer_order:
            print(f"\n[4/5] SKIPPING {stage_name} (no layers specified)")
            continue

        # Which layers in this stage to skip (already trained)
        stage_layer_indices = {li for li, _ in layer_order}
        stage_skip = resume_skip & stage_layer_indices

        print(f"\n[4/5] Running {stage_name} ({len(layer_order)} layers)...")
        stage_results, last_acc = progressive_stage(
            model=model,
            teacher=teacher,
            train_ds=train_ds,
            eval_ds=eval_ds,
            args=args,
            stage_name=stage_name,
            layer_order=layer_order,
            replaced_so_far=replaced_so_far,
            output_dir=output_dir,
            num_layers=num_layers,
            lpan_acc=lpan_acc,
            device=device,
            skip_layers=stage_skip,
        )
        all_layer_results.extend(stage_results)

    # Save progressive checkpoint
    prog_path = output_dir / "progressive_final" / "best_model"
    prog_path.mkdir(parents=True, exist_ok=True)
    for p in model.parameters():
        p.data = p.data.contiguous()
    model.save_pretrained(str(prog_path))
    print(f"\n  Progressive checkpoint saved ({last_acc:.4f})")

    # ── 5. Global fine-tune ───────────────────────────────────────────────
    final_acc = last_acc
    if not args.skip_global_finetune:
        global_lr = args.lr / 3
        print(f"\n[5/5] Global fine-tune ({args.final_epochs} epochs, lr={global_lr:.2e})...")

        trainable = freeze_for_global_finetune(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

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

        # Cleanup
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

    layer_summary = summarize_attention_types(model)

    results = {
        "model": args.model,
        "task": args.task,
        "max_seq_len": args.max_seq_len,
        "lpan_accuracy": lpan_acc,
        "final_accuracy": final_acc,
        "accuracy_drop_from_lpan": (lpan_acc - final_acc) * 100,
        "layer_composition": {
            "linear_mixing": linear_mixing_layers,
            "quad": quad_attention_layers,
            "lpan": lpan_layers,
        },
        "final_layer_types": layer_summary,
        "per_layer_results": all_layer_results,
        "epochs_per_layer": args.epochs_per_layer,
        "final_epochs": args.final_epochs,
        "lr": args.lr,
        "gamma": args.gamma,
        "gamma_decay": args.gamma_decay,
        "lr_schedule": args.lr_schedule,
        "stage_order": args.stage_order,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    drop = (lpan_acc - final_acc) * 100
    print(f"\n{'='*70}")
    print(f"  Results Summary — HyPER-LPAN")
    print(f"{'='*70}")
    print(f"  LPAN baseline:      {lpan_acc:.4f}")
    for r in all_layer_results:
        print(f"  After L{r['layer']:2d} ({r['layer_type']:13s}): "
              f"{r['accuracy']:.4f}  ({r['drop_from_lpan']:+.2f}%)")
    if not args.skip_global_finetune:
        print(f"  Global fine-tune:   {final_acc:.4f}")
    print(f"  Drop from LPAN:     {drop:+.2f}%")
    print(f"  Final layer types:  {layer_summary}")
    print(f"  Saved to: {final_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
