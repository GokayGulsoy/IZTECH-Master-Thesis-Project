#!/usr/bin/env python3
"""Staged LPAN: progressive polynomial replacement for FHE-compatible BERT.

Pipeline (works for any BERT variant in MODEL_REGISTRY):
  Stage 1: Replace GELU → learnable polynomial, fine-tune with CE loss
  Stage 2: Replace Softmax → learnable polynomial, fine-tune with CE loss
  Stage 3: Replace LayerNorm → learnable polynomial, fine-tune with KD loss

LayerNorm replacement uses Knowledge Distillation because CE alone fails —
when LN is replaced, every subsequent layer receives mis-scaled inputs,
creating a loss landscape trap. KD with the Stage-2 model as teacher
provides the gradient signal needed to escape.

Usage:
  python run_staged_lpan.py --model base          # BERT-Base
  python run_staged_lpan.py --model tiny           # BERT-Tiny
  python run_staged_lpan.py --model mini small     # multiple models
  python run_staged_lpan.py --model all            # all four variants
"""
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

# Suppress HuggingFace/transformers noise that floods the log:
#   - "A parameter name that contains `beta`/`gamma` will be renamed..."
#     (old-style BERT checkpoint key renaming; cosmetic, no effect on weights)
#   - "Some weights of the model checkpoint ... were not used..."
#     (expected: from_pretrained ignores our custom poly_softmax/act_fn keys)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import torch
from datasets import load_dataset
from safetensors.torch import load_file as _load_safetensors
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).parent))
from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR
from fhe_thesis.models.profiling import profile_model, compute_poly_coefficients
from fhe_thesis.models.replacement import replace_activations
from fhe_thesis.training.trainer import (
    NaNSafeTrainer,
    attn_distill_and_eval,
    compute_metrics,
    detect_device,
    distill_and_eval,
)


def _restore_poly_coeffs(model: torch.nn.Module, checkpoint_path) -> int:
    """Copy any 'coeffs' tensors from a safetensors/bin checkpoint into model.

    from_pretrained silently ignores custom submodule keys (polynomial
    activation coefficients are not in the base BERT architecture), so we
    must manually patch them back after every checkpoint load.
    Works for both learnable (nn.Parameter) and frozen (buffer) coefficients.
    Returns the number of coefficient tensors restored.
    """
    sf = Path(checkpoint_path) / "model.safetensors"
    ckpt = Path(checkpoint_path) / "pytorch_model.bin"
    if sf.exists():
        state = _load_safetensors(str(sf))
    elif ckpt.exists():
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    else:
        return 0
    # Combine parameters and buffers — coeffs are Parameters when learnable=True
    # but registered as buffers when learnable=False.
    all_tensors = dict(model.named_parameters())
    all_tensors.update(dict(model.named_buffers()))
    n = 0
    for k, v in state.items():
        if "coeffs" in k and k in all_tensors:
            target = all_tensors[k]
            if target.shape == v.shape:
                target.data.copy_(v.to(target.device))
                n += 1
            elif v.dim() == 1 and target.dim() == 2:
                # Old checkpoint (shared) → new model (per-head): broadcast
                target.data.copy_(
                    v.unsqueeze(0).expand_as(target).to(target.device)
                )
                n += 1
            else:
                print(f"  Warning: shape mismatch for {k}: "
                      f"checkpoint {v.shape} vs model {target.shape}, skipping")
    del state
    return n


def _freeze_for_ln_stage(model: torch.nn.Module) -> int:
    """Freeze Stage-2 weights and train only LN replacements plus final head.

    Stage 3 is meant to adapt the newly inserted polynomial LayerNorm modules.
    Leaving the whole network trainable makes Base unstable because KD updates
    can push the already-good Stage-2 solution far away in a single epoch.
    """
    trainable = 0
    for name, param in model.named_parameters():
        should_train = (
            ".attention.output.LayerNorm." in name
            or ".output.LayerNorm." in name
            or name.startswith("bert.pooler.")
            or name.startswith("classifier.")
        )
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
    return trainable


def _freeze_for_progressive_ln(model: torch.nn.Module, layer_idx: int, replaced_layers: list) -> int:
    """Freeze everything except LN coefficients for replaced layers + classifier.

    Unlike the single-layer variant, this keeps ALL already-replaced LN layers
    trainable so they can co-adapt as new polynomial LNs are added.  Early
    layers (L0, L1, ...) suffer large accuracy drops because their polynomial
    LN approximation error compounds through all downstream layers.  Allowing
    previously-replaced LNs to keep updating lets them compensate for the
    perturbation introduced by each new polynomial LN.
    """
    trainable = 0
    for name, param in model.named_parameters():
        is_replaced_ln = False
        for prev_li in replaced_layers:
            prefix = f"bert.encoder.layer.{prev_li}."
            if name.startswith(prefix) and "LayerNorm" in name:
                is_replaced_ln = True
                break
        is_head = (
            name.startswith("bert.pooler.")
            or name.startswith("classifier.")
        )
        should_train = is_replaced_ln or is_head
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
    return trainable


def _set_reproducibility(seed: int) -> None:
    """Best-effort reproducibility for thesis experiments."""
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# GLUE tasks used by SOTA FHE papers (THE-X, Iron, MPCFormer, BOLT)
SOTA_GLUE_TASKS = ["sst2", "qnli", "qqp", "mrpc"]

# Column mapping for each supported GLUE task
TASK_CONFIG = {
    "sst2": {"text_a": "sentence",   "text_b": None},
    "qnli": {"text_a": "question",   "text_b": "sentence"},
    "qqp":  {"text_a": "question1",  "text_b": "question2"},
    "mrpc": {"text_a": "sentence1",  "text_b": "sentence2"},
}


def load_data(model_name: str, task: str = "sst2"):
    """Load and tokenize a GLUE task dataset (sst2/qnli/qqp/mrpc)."""
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task '{task}'. Choose from: {list(TASK_CONFIG.keys())}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", task)
    col_a = TASK_CONFIG[task]["text_a"]
    col_b = TASK_CONFIG[task]["text_b"]

    def tokenize_fn(ex):
        if col_b is None:
            return tokenizer(
                ex[col_a], truncation=True, padding="max_length", max_length=128
            )
        return tokenizer(
            ex[col_a], ex[col_b], truncation=True, padding="max_length", max_length=128
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized["train"], tokenized["validation"]


def run_ce_stage(
    model, train_ds, eval_ds, poly_coeffs, hidden, replace_types,
    stage_name, stage_output, epochs, bs, lr, device, seed,
):
    """Run a CE-loss fine-tuning stage (for GELU / Softmax replacement)."""
    replace_activations(
        model, poly_coeffs, hidden, learnable=True, replace_types=replace_types
    )

    poly_params = sum(p.numel() for n, p in model.named_parameters() if "coeffs" in n)
    print(f"  Learnable poly coefficients so far: {poly_params}")

    # Zero-shot eval after replacement
    zs_args = TrainingArguments(
        output_dir=stage_output + "_zs", per_device_eval_batch_size=32,
        report_to="none", disable_tqdm=True,
        seed=seed,
        data_seed=seed,
    )
    zs_trainer = Trainer(
        model=model, args=zs_args, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    zs_acc = zs_trainer.evaluate()["eval_accuracy"]
    print(f"  After {stage_name} replacement (before FT): {zs_acc:.4f}")

    # Fine-tune
    args = TrainingArguments(
        output_dir=stage_output,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        seed=seed,
        data_seed=seed,
    )

    trainer = NaNSafeTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print(f"  Training {stage_name} for {epochs} epochs (bs={bs}, lr={lr})...")
    trainer.train()
    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"  {stage_name} accuracy: {acc:.4f} ({acc:.2%})")

    # Save stage model
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        trainer.save_model(os.path.join(stage_output, "best_model"))
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    return acc


def run_attn_kd_stage(
    model, train_ds, eval_ds, poly_coeffs, hidden,
    stage_output, epochs, bs, lr, device, stage1_path, seed,
):
    """Run Softmax replacement with attention-level knowledge distillation.

    Teacher = Stage 1 model (polynomial GELU + original Softmax).
    Student = Stage 1 model with Softmax replaced by polynomial.
    Loss = alpha*CE + beta*KL(teacher_attn||student_attn) + gamma*MSE(hiddens)
    """
    # --- Load teacher: Stage 1 model with original Softmax ---
    print("  Loading Stage 1 model as teacher (original Softmax)...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(stage1_path), num_labels=2
    )
    # Teacher keeps original Softmax — only apply GELU replacement
    replace_activations(
        teacher, poly_coeffs, hidden, learnable=False,
        replace_types=["GELU"],
    )
    n = _restore_poly_coeffs(teacher, stage1_path)
    print(f"  Teacher: poly GELU restored ({n} tensors), original Softmax kept.")
    teacher.to(device)
    teacher.eval()

    # --- Replace Softmax in student ---
    replace_activations(
        model, poly_coeffs, hidden, learnable=True, replace_types=["Softmax"]
    )

    poly_params = sum(p.numel() for n, p in model.named_parameters() if "coeffs" in n)
    print(f"  Learnable poly coefficients so far: {poly_params}")

    # Zero-shot eval after replacement
    zs_args = TrainingArguments(
        output_dir=stage_output + "_zs", per_device_eval_batch_size=32,
        report_to="none", disable_tqdm=True,
        seed=seed, data_seed=seed,
    )
    zs_trainer = Trainer(
        model=model, args=zs_args, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    zs_acc = zs_trainer.evaluate()["eval_accuracy"]
    print(f"  After Softmax replacement (before FT): {zs_acc:.4f}")

    # Fine-tune with attention-level KD
    ft_res = attn_distill_and_eval(
        student_model=model,
        teacher_model=teacher,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=stage_output,
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        label="Stage 2 (Softmax, AttnKD)",
        use_fp16=True,
        max_grad_norm=1.0,
        alpha=1.0,   # CE task loss
        beta=0.01,   # attention KL (scaled down — raw KL is O(100))
        gamma=10.0,  # hidden state MSE (scaled up — raw MSE is O(0.01))
        seed=seed,
    )

    acc = ft_res["accuracy"]

    # Save stage model
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        import shutil
        best_path = os.path.join(stage_output, "best_model")
        if not os.path.exists(best_path):
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    del teacher
    torch.cuda.empty_cache()
    return acc


def run_progressive_softmax_stage(
    model, train_ds, eval_ds, poly_coeffs, hidden, num_layers,
    stage_output, epochs_per_layer, bs, lr, device, stage1_path, seed,
    start_layer: int = 0,
):
    """Replace Softmax layer-by-layer with AttnKD fine-tuning after each.

    Instead of replacing all Softmax layers at once (causing compound error),
    this replaces one layer's Softmax at a time and fine-tunes, so each
    polynomial adapts while remaining layers still provide stable gradients.

    Uses two literature-inspired strategies to mitigate last-layer collapse:
      1. Depth-adaptive epochs: later layers (closer to classifier) get more
         training epochs because their errors have no downstream correction.
         (Inspired by curriculum learning and gradual unfreezing from ULMFiT.)
      2. Layer-wise LR scaling: later layers get higher LR since they need
         larger parameter updates to match the classifier's expectations.
         (Inspired by LLRD from ELECTRA/DeBERTa fine-tuning.)

    Teacher = Stage 1 model (polynomial GELU + original Softmax).
    """
    # --- Load teacher once: Stage 1 model with original Softmax ---
    print("  Loading Stage 1 model as teacher (original Softmax)...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(stage1_path), num_labels=2
    )
    replace_activations(
        teacher, poly_coeffs, hidden, learnable=False,
        replace_types=["GELU"],
    )
    n = _restore_poly_coeffs(teacher, stage1_path)
    print(f"  Teacher: poly GELU restored ({n} tensors), original Softmax kept.")
    teacher.to(device)
    teacher.eval()

    acc = None
    # If resuming, restore already-trained Softmax layers from checkpoints
    if start_layer > 0:
        print(f"  Resuming from layer {start_layer} — restoring layers 0..{start_layer-1} from checkpoints")
        for prev_li in range(start_layer):
            replace_activations(
                model, poly_coeffs, hidden, learnable=False,
                replace_types=["Softmax"], layer_indices=[prev_li],
            )
            prev_ckpt = os.path.join(stage_output, f"layer_{prev_li}", "best_model")
            if os.path.isdir(prev_ckpt):
                n = _restore_poly_coeffs(model, prev_ckpt)
                print(f"    L{prev_li}: restored {n} coeff tensors from checkpoint")
            else:
                print(f"    L{prev_li}: WARNING — no best_model checkpoint found at {prev_ckpt}")

    for li in range(start_layer, num_layers):
        # Depth-adaptive epochs: last layer gets 2× epochs,
        # second-to-last gets 1.5×, rest get base epochs.
        remaining = num_layers - 1 - li  # 0 for last layer
        if remaining == 0:
            layer_epochs = epochs_per_layer * 2
        elif remaining == 1:
            layer_epochs = max(epochs_per_layer, int(epochs_per_layer * 1.5))
        else:
            layer_epochs = epochs_per_layer

        # Layer-wise LR scaling: later layers get higher LR (LLRD-inspired).
        # Scale factor: 1.0 for layer 0, up to 1.5 for the last layer.
        lr_scale = 1.0 + 0.5 * (li / max(1, num_layers - 1))
        layer_lr = lr * lr_scale

        print(f"\n  --- Stage 2 sub-step {li+1}/{num_layers}: "
              f"Replace Softmax in layer {li} "
              f"(epochs={layer_epochs}, lr={layer_lr:.1e}) ---")

        # Replace Softmax in this single layer only
        replace_activations(
            model, poly_coeffs, hidden, learnable=True,
            replace_types=["Softmax"], layer_indices=[li],
        )

        poly_params = sum(
            p.numel() for pn, p in model.named_parameters() if "coeffs" in pn
        )
        print(f"  Learnable poly coefficients so far: {poly_params}")

        # Zero-shot eval after this layer's replacement
        zs_args = TrainingArguments(
            output_dir=stage_output + f"_L{li}_zs",
            per_device_eval_batch_size=32,
            report_to="none", disable_tqdm=True,
            seed=seed, data_seed=seed,
        )
        zs_trainer = Trainer(
            model=model, args=zs_args, eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        zs_acc = zs_trainer.evaluate()["eval_accuracy"]
        print(f"  After L{li} Softmax replacement (before FT): {zs_acc:.4f}")

        # Fine-tune with AttnKD
        sub_output = os.path.join(stage_output, f"layer_{li}")
        ft_res = attn_distill_and_eval(
            student_model=model,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=sub_output,
            epochs=layer_epochs,
            batch_size=bs,
            lr=layer_lr,
            label=f"Stage 2 L{li} Softmax (AttnKD)",
            use_fp16=True,
            max_grad_norm=1.0,
            alpha=1.0,
            beta=0.01,
            gamma=10.0,
            seed=seed,
        )
        acc = ft_res["accuracy"]
        print(f"  After L{li} fine-tune: {acc:.4f} ({acc:.2%})")
        # Remove intermediate layer checkpoint to keep disk usage low
        import shutil
        shutil.rmtree(sub_output, ignore_errors=True)

    # Save final Stage 2 model
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        best_path = os.path.join(stage_output, "best_model")
        os.makedirs(best_path, exist_ok=True)
        model.save_pretrained(best_path)
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    del teacher
    torch.cuda.empty_cache()
    return acc


def run_kd_stage(
    student, train_ds, eval_ds, poly_coeffs, hidden,
    stage_output, epochs, bs, lr, device, stage2_path, seed,
):
    """Run KD-loss fine-tuning stage for LayerNorm replacement."""
    # Load teacher (Stage 2 model with standard LN)
    print("  Loading Stage 2 model as teacher...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(stage2_path), num_labels=2
    )
    replace_activations(
        teacher, poly_coeffs, hidden, learnable=False,
        replace_types=["GELU", "Softmax"],
    )
    # Restore trained polynomial coefficients from Stage 2 checkpoint.
    # from_pretrained skips custom submodule keys (poly_softmax, act_fn.coeffs),
    # so replace_activations above gives profiling-based initial values.
    # We overwrite them here with the trained values to avoid teacher/student mismatch.
    n = _restore_poly_coeffs(teacher, stage2_path)
    print(f"  Teacher poly coeffs restored from Stage 2 checkpoint ({n} tensors).")
    teacher.to(device)
    teacher.eval()

    # Replace LN in student
    replace_activations(
        student, poly_coeffs, hidden, learnable=True, replace_types=["LN"]
    )
    trainable_params = _freeze_for_ln_stage(student)

    poly_params = sum(
        p.numel() for n, p in student.named_parameters() if "coeffs" in n
    )
    print(f"  Learnable poly coefficients: {poly_params}")
    print(f"  Total trainable params in Stage 3: {trainable_params}")

    # Zero-shot eval after LN replacement
    zs_args = TrainingArguments(
        output_dir=stage_output + "_zs", per_device_eval_batch_size=32,
        report_to="none", disable_tqdm=True,
        seed=seed,
        data_seed=seed,
    )
    zs_trainer = Trainer(
        model=student, args=zs_args, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    zs_acc = zs_trainer.evaluate()["eval_accuracy"]
    print(f"  After LN replacement (before FT): {zs_acc:.4f}")

    # KD fine-tuning
    # Use lower LR for large models to prevent divergence
    ft_res = distill_and_eval(
        student_model=student,
        teacher_model=teacher,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=stage_output,
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        label="Stage 3 LN (KD)",
        use_fp16=True,
        alpha=0.5,
        temperature=4.0,
        max_grad_norm=1.0,
        seed=seed,
    )

    del teacher
    torch.cuda.empty_cache()
    return ft_res["accuracy"], poly_params


def run_progressive_ln_stage(
    model, train_ds, eval_ds, poly_coeffs, hidden, num_layers,
    stage_output, epochs_per_layer, bs, lr, device,
    stage2_path, seed, start_layer=0,
):
    """Progressive layer-by-layer LayerNorm replacement with AttnKD.

    Mirrors run_progressive_softmax_stage: replaces one layer's LN at a time
    and fine-tunes with attention-level KD, preventing the compound error that
    causes all-at-once LN replacement to collapse on deep models (Base: 24 LNs).

    Teacher = Stage 2 model (polynomial GELU + Softmax, original LN).
    """
    # --- Load teacher once: Stage 2 model with original LN ---
    print("  Loading Stage 2 model as teacher (original LN)...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        str(stage2_path), num_labels=2
    )
    replace_activations(
        teacher, poly_coeffs, hidden, learnable=False,
        replace_types=["GELU", "Softmax"],
    )
    n = _restore_poly_coeffs(teacher, stage2_path)
    print(f"  Teacher: poly GELU+Softmax restored ({n} tensors), original LN kept.")
    teacher.to(device)
    teacher.eval()

    acc = None
    # If resuming, restore full model state from the last completed layer's
    # checkpoint (includes classifier, pooler, and all weights that adapted
    # during previous sub-steps), then re-create polynomial LN modules and
    # restore their coefficients for all previously-replaced layers.
    if start_layer > 0:
        last_ckpt = os.path.join(stage_output, f"layer_{start_layer-1}", "best_model")
        print(f"  Resuming from layer {start_layer} — loading full model from {last_ckpt}")
        if os.path.isdir(last_ckpt):
            # Load full model weights (classifier, pooler, embeddings, etc.)
            sf = Path(last_ckpt) / "model.safetensors"
            ckpt_bin = Path(last_ckpt) / "pytorch_model.bin"
            if sf.exists():
                full_state = _load_safetensors(str(sf))
            elif ckpt_bin.exists():
                full_state = torch.load(str(ckpt_bin), map_location="cpu", weights_only=False)
            else:
                full_state = None
                print(f"    WARNING: no checkpoint file found in {last_ckpt}")
            if full_state is not None:
                # Load non-coeffs weights (coeffs will be loaded after
                # polynomial modules are created)
                model_state = model.state_dict()
                loaded = 0
                for k, v in full_state.items():
                    if "coeffs" not in k and k in model_state and model_state[k].shape == v.shape:
                        model_state[k] = v
                        loaded += 1
                model.load_state_dict(model_state, strict=False)
                print(f"    Loaded {loaded} non-coeff tensors (classifier, pooler, etc.)")
                del full_state

        # Now create polynomial LN modules and restore their coefficients
        for prev_li in range(start_layer):
            replace_activations(
                model, poly_coeffs, hidden, learnable=True,
                replace_types=["LN"], layer_indices=[prev_li],
            )
            prev_ckpt = os.path.join(stage_output, f"layer_{prev_li}", "best_model")
            if os.path.isdir(prev_ckpt):
                n = _restore_poly_coeffs(model, prev_ckpt)
                print(f"    L{prev_li}: restored {n} coeff tensors from checkpoint")
            else:
                print(f"    L{prev_li}: WARNING — no best_model checkpoint found at {prev_ckpt}")

    replaced_so_far = list(range(start_layer))  # track which layers have poly LN

    for li in range(start_layer, num_layers):
        # Depth-adaptive epochs:
        #   - Early layers (first 1/3): 2× base — compound error needs more training.
        #   - Middle layers: base epochs (recovery happens in 1-2 epochs).
        #   - Last two layers: 1.5× base (close to classifier).
        remaining = num_layers - 1 - li
        early_boundary = num_layers // 3  # first 4 layers for 12-layer model
        if li < early_boundary:
            layer_epochs = epochs_per_layer * 2
        elif remaining <= 1:
            layer_epochs = max(epochs_per_layer, int(epochs_per_layer * 1.5))
        else:
            layer_epochs = epochs_per_layer

        # Layer-wise LR scaling: later layers get higher LR (LLRD-inspired).
        lr_scale = 1.0 + 0.5 * (li / max(1, num_layers - 1))
        layer_lr = lr * lr_scale

        print(f"\n  --- Stage 3 sub-step {li+1}/{num_layers}: "
              f"Replace LN in layer {li} "
              f"(epochs={layer_epochs}, lr={layer_lr:.1e}) ---")

        # Replace LN in this single layer only
        replace_activations(
            model, poly_coeffs, hidden, learnable=True,
            replace_types=["LN"], layer_indices=[li],
        )
        replaced_so_far.append(li)

        # Freeze everything except ALL replaced LN layers + classifier
        # (co-adaptation: previous poly LNs keep updating to compensate)
        trainable = _freeze_for_progressive_ln(model, li, replaced_so_far)

        poly_params = sum(
            p.numel() for pn, p in model.named_parameters() if "coeffs" in pn
        )
        print(f"  Learnable poly coefficients so far: {poly_params}")
        print(f"  Trainable params this sub-step: {trainable}")

        # Zero-shot eval after this layer's replacement
        zs_args = TrainingArguments(
            output_dir=stage_output + f"_L{li}_zs",
            per_device_eval_batch_size=32,
            report_to="none", disable_tqdm=True,
            seed=seed, data_seed=seed,
        )
        zs_trainer = Trainer(
            model=model, args=zs_args, eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        zs_acc = zs_trainer.evaluate()["eval_accuracy"]
        print(f"  After L{li} LN replacement (before FT): {zs_acc:.4f}")

        # Fine-tune with AttnKD
        # fp16 disabled: polynomial LN coefficients are tiny (1e-3 to 1e-6),
        # and fp16 gradient scaling can collapse to zero after a single NaN
        # event, permanently killing training.  fp32 is ~2× slower per step
        # but avoids the gradient underflow that killed v1 and v2-fp16.
        sub_output = os.path.join(stage_output, f"layer_{li}")
        ft_res = attn_distill_and_eval(
            student_model=model,
            teacher_model=teacher,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=sub_output,
            epochs=layer_epochs,
            batch_size=bs,
            lr=layer_lr,
            label=f"Stage 3 L{li} LN (AttnKD)",
            use_fp16=False,
            max_grad_norm=1.0,
            alpha=1.0,
            beta=0.01,
            gamma=10.0,
            seed=seed,
            lr_scheduler_type="constant_with_warmup",
        )
        acc = ft_res["accuracy"]
        print(f"  After L{li} LN fine-tune: {acc:.4f} ({acc:.2%})")
        # Remove intermediate layer checkpoint to keep disk usage low
        import shutil
        shutil.rmtree(sub_output, ignore_errors=True)

    # Save final Stage 3 model
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        best_path = os.path.join(stage_output, "best_model")
        os.makedirs(best_path, exist_ok=True)
        model.save_pretrained(best_path)
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    del teacher
    torch.cuda.empty_cache()

    poly_params = sum(
        p.numel() for pn, p in model.named_parameters() if "coeffs" in pn
    )
    return acc, poly_params


def _train_baseline(model_name: str, task: str, result_dir: Path,
                    bs: int, lr: float, device, seed: int) -> float:
    """Fine-tune a baseline model on a GLUE task and save to result_dir/baseline/best_model."""
    from fhe_thesis.training.trainer import NaNSafeTrainer, compute_metrics
    baseline_save = str(result_dir / "baseline" / "best_model")
    os.makedirs(baseline_save, exist_ok=True)
    print(f"  Fine-tuning baseline on '{task}'...")
    train_ds, eval_ds = load_data(model_name, task)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    args = TrainingArguments(
        output_dir=str(result_dir / "baseline"),
        num_train_epochs=3,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        logging_steps=100,
    )
    trainer = NaNSafeTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    acc = metrics.get("eval_accuracy", 0.0)
    model.save_pretrained(baseline_save)
    # Persist accuracy
    with open(result_dir / "results.json", "w") as f:
        json.dump({"baseline_acc": acc, "task": task}, f, indent=2)
    print(f"  Baseline accuracy on {task}: {acc:.4f}")
    return acc


def run_staged_lpan(model_key: str, task: str = "sst2", degree: int = 8,
                    start_stage: int = 1, seed: int = 42,
                    start_layer: int = 0, start_ln_layer: int = 0):
    """Run the full 3-stage LPAN pipeline for one model variant on a GLUE task."""
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["name"]
    short = cfg["short"]
    num_layers = cfg["layers"]
    hidden = cfg["hidden"]
    bs = cfg["batch_size"]
    lr = cfg["lr"]

    # Task-scoped output directory: results/multi_model/<task>/<model_key>
    result_dir = MULTI_MODEL_DIR / task / model_key
    result_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = result_dir / "baseline" / "best_model"

    device = detect_device()

    # Auto-train baseline if it doesn't exist for this task
    results_file = result_dir / "results.json"
    baseline_acc = 0.0
    if not baseline_path.exists():
        print(f"  Baseline not found for task='{task}', model='{model_key}' — training now...")
        baseline_acc = _train_baseline(model_name, task, result_dir, bs, lr, device, seed)
    else:
        if results_file.exists():
            with open(results_file) as f:
                prev = json.load(f)
            baseline_acc = prev.get("baseline_acc", 0.0)

    print(f"\n{'='*70}")
    print(f"  {short} — Staged LPAN Pipeline")
    print(f"  Baseline: {baseline_acc:.4f} ({baseline_acc:.2%})")
    print(f"  Stages: GELU(CE) → Softmax(Progressive AttnKD) → LayerNorm(KD)")
    print(f"  Degree: {degree} (adaptive per operation/depth)")
    print(f"  Task:  {task}")
    print(f"  Seed: {seed}")
    print(f"{'='*70}")

    _set_reproducibility(seed)

    # Load data
    print("\n[0/5] Loading data...")
    train_ds, eval_ds = load_data(model_name, task)

    # Profile activations
    print("\n[1/5] Profiling activations...")
    profile_data = profile_model(model_name, num_layers, 1000)
    poly_coeffs = compute_poly_coefficients(profile_data, num_layers, degree)

    # Load baseline model
    model = AutoModelForSequenceClassification.from_pretrained(
        str(baseline_path), num_labels=2
    )
    model.to(device)  # device already set above

    # ── Stage 1: GELU (CE) ──
    s1_output = str(result_dir / "staged_lpan_s1_gelu")
    s1_model_path = Path(s1_output) / "best_model"
    if start_stage <= 1:
        print(f"\n{'='*70}")
        print(f"  Stage 1/3: Replace GELU → Learnable Polynomial (CE)")
        print(f"{'='*70}")
        s1_acc = run_ce_stage(
            model, train_ds, eval_ds, poly_coeffs, hidden,
            replace_types=["GELU"],
            stage_name="Stage 1 (GELU)",
            stage_output=s1_output,
            epochs=3, bs=bs, lr=lr, device=device, seed=seed,
        )
    else:
        # Load Stage 1 model, re-apply GELU replacement, restore trained coefficients.
        # from_pretrained ignores custom 'intermediate_act_fn.coeffs' keys, so we must
        # call replace_activations + _restore_poly_coeffs to get the trained polynomial.
        print(f"\n  Skipping Stage 1 — loading from {s1_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(s1_model_path), num_labels=2
        )
        replace_activations(
            model, poly_coeffs, hidden, learnable=False, replace_types=["GELU"]
        )
        n_restored = _restore_poly_coeffs(model, s1_model_path)
        print(f"  Stage 1 GELU: poly modules created, {n_restored} coeff tensors restored from checkpoint.")
        model.to(device)
        s1_acc = 0.0  # will be read from previous results if available

    # ── Stage 2: Softmax (Attention KD) ──
    s2_output = str(result_dir / "staged_lpan_s2_softmax")
    s2_model_path = Path(s2_output) / "best_model"
    if start_stage <= 2:
        print(f"\n{'='*70}")
        print(f"  Stage 2/3: Replace Softmax → Learnable Polynomial (Progressive AttnKD)")
        print(f"  Replacing layer-by-layer: {num_layers} layers, depth-adaptive epochs + LR scaling")
        print(f"{'='*70}")
        s2_acc = run_progressive_softmax_stage(
            model, train_ds, eval_ds, poly_coeffs, hidden, num_layers,
            stage_output=s2_output,
            epochs_per_layer=2, bs=bs, lr=lr, device=device,
            stage1_path=s1_model_path, seed=seed,
            start_layer=start_layer,
        )
    else:
        # Load Stage 2 model
        print(f"\n  Skipping Stage 2 — loading from {s2_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(s2_model_path), num_labels=2
        )
        # Re-create polynomial modules for GELU and Softmax, then restore
        # trained coefficients from their respective stage checkpoints.
        replace_activations(
            model, poly_coeffs, hidden, learnable=False,
            replace_types=["GELU", "Softmax"],
        )
        n1 = _restore_poly_coeffs(model, s1_model_path)  # GELU coeffs from Stage 1
        n2 = _restore_poly_coeffs(model, s2_model_path)  # Softmax coeffs from Stage 2
        print(f"  Stage 1 GELU: {n1} coeff tensors restored from Stage 1 checkpoint.")
        print(f"  Stage 2 Softmax: {n2} coeff tensors restored from Stage 2 checkpoint.")
        model.to(device)
        s2_acc = 0.0

    # ── Stage 3: LayerNorm ──
    stage2_model_path = Path(s2_output) / "best_model"

    if num_layers >= 12:
        # Progressive LN replacement with AttnKD for deep models (Base)
        s3_output = str(result_dir / "staged_lpan_s3_ln_progressive")
        s3_lr = 5e-6
        print(f"\n{'='*70}")
        print(f"  Stage 3/3: Replace LayerNorm → Learnable Polynomial (Progressive AttnKD)")
        print(f"  Replacing layer-by-layer: {num_layers} layers, depth-adaptive epochs + LR scaling")
        print(f"{'='*70}")
        s3_acc, poly_params = run_progressive_ln_stage(
            model, train_ds, eval_ds, poly_coeffs, hidden, num_layers,
            stage_output=s3_output,
            epochs_per_layer=2, bs=bs, lr=s3_lr, device=device,
            stage2_path=stage2_model_path, seed=seed,
            start_layer=start_ln_layer,
        )
    else:
        # All-at-once LN replacement with vanilla KD for small models
        s3_output = str(result_dir / "staged_lpan_s3_ln_kd")
        s3_lr = lr * 0.5
        s3_epochs = 5
        print(f"\n{'='*70}")
        print(f"  Stage 3/3: Replace LayerNorm → Learnable Polynomial (KD)")
        print(f"{'='*70}")
        s3_acc, poly_params = run_kd_stage(
            student=model,
            train_ds=train_ds, eval_ds=eval_ds,
            poly_coeffs=poly_coeffs, hidden=hidden,
            stage_output=s3_output,
            epochs=s3_epochs, bs=bs, lr=s3_lr, device=device,
            stage2_path=stage2_model_path, seed=seed,
        )

    # ── Compute multiplicative depth ──
    total_depth = 0
    for li in range(num_layers):
        for op in ["GELU", "Softmax", "LN"]:
            k = f"L{li}_{op}"
            if k in poly_coeffs:
                d = poly_coeffs[k]["degree"]
                total_depth += max(1, math.ceil(math.log2(d + 1)))

    # ── Final Summary ──
    drop = (baseline_acc - s3_acc) * 100
    print(f"\n{'='*70}")
    print(f"  {short} — STAGED LPAN RESULTS")
    print(f"{'='*70}")
    print(f"  Baseline:              {baseline_acc:.4f} ({baseline_acc:.2%})")
    print(f"  Stage 1 (GELU, CE):    {s1_acc:.4f} ({s1_acc:.2%})")
    print(f"  Stage 2 (Softmax,KD):  {s2_acc:.4f} ({s2_acc:.2%})")
    print(f"  Stage 3 (LN, KD):     {s3_acc:.4f} ({s3_acc:.2%})")
    print(f"  Drop from baseline:    {drop:.2f}%")
    print(f"  Depth:                 {total_depth}")
    print(f"  Poly params:           {poly_params}")
    print(f"{'='*70}")

    # Save final model
    final_path = str(result_dir / "staged_lpan_final" / "best_model")
    os.makedirs(final_path, exist_ok=True)
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        model.save_pretrained(final_path)
        print(f"  Final model saved to: {final_path}")
    except Exception as e:
        print(f"  Warning: Could not save final model: {e}")

    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "short": short,
        "task": task,
        "layers": num_layers,
        "hidden": hidden,
        "params_m": cfg["params_m"],
        "baseline_acc": baseline_acc,
        "stage1_acc": s1_acc,
        "stage2_acc": s2_acc,
        "stage3_acc": s3_acc,
        "accuracy_drop_pct": drop,
        "poly_degree": degree,
        "seed": seed,
        "total_depth": total_depth,
        "poly_params": poly_params,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Staged LPAN: progressive polynomial replacement for FHE-compatible BERT"
    )
    parser.add_argument(
        "--model", nargs="+", default=["base"],
        help="Model key(s) from MODEL_REGISTRY, or 'all'",
    )
    parser.add_argument(
        "--task", nargs="+", default=["sst2"],
        help=f"GLUE task(s) to run, or 'sota' for all SOTA tasks {SOTA_GLUE_TASKS}",
    )
    parser.add_argument("--degree", type=int, default=8, help="Base polynomial degree")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--start-stage", type=int, default=1, choices=[1, 2, 3],
        help="Stage to start from (default: 1). Skipped stages load from checkpoints.",
    )
    parser.add_argument(
        "--start-layer", type=int, default=0,
        help="Layer to resume from in Stage 2 progressive Softmax (default: 0).",
    )
    parser.add_argument(
        "--start-ln-layer", type=int, default=0,
        help="Layer to resume from in Stage 3 progressive LN (default: 0).",
    )
    args = parser.parse_args()

    model_keys = list(MODEL_REGISTRY.keys()) if "all" in args.model else args.model
    for k in model_keys:
        if k not in MODEL_REGISTRY:
            print(f"Unknown model key: {k}. Options: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    task_keys = SOTA_GLUE_TASKS if "sota" in args.task else args.task
    for t in task_keys:
        if t not in TASK_CONFIG:
            print(f"Unknown task: {t}. Options: {list(TASK_CONFIG.keys())}")
            sys.exit(1)

    all_results = []
    for task in task_keys:
        for model_key in model_keys:
            print(f"\n{'#'*70}")
            print(f"  Task: {task.upper()}  |  Model: {model_key}")
            print(f"{'#'*70}")
            result = run_staged_lpan(
                model_key, task, args.degree, args.start_stage,
                args.seed, args.start_layer, args.start_ln_layer
            )
            if result:
                all_results.append(result)

    # Save combined results scoped by task
    out_file = MULTI_MODEL_DIR / f"staged_lpan_results_{'_'.join(task_keys)}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
