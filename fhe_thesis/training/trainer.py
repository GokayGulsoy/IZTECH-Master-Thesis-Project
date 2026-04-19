"""Training utilities: metrics, data loading, fine-tuning helpers."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from fhe_thesis.tasks import GLUE_TASKS, TaskConfig, get_task


class NaNSafeTrainer(Trainer):
    """Trainer that sanitises NaN/Inf in gradients to prevent cascade corruption.

    When clip_grad_norm_ encounters ANY NaN gradient, it computes
    total_norm = NaN, then multiplies ALL gradients by NaN, permanently
    corrupting every parameter in a single step.

    Fix: replace NaN/Inf values within each gradient with 0 in-place.
    This preserves good gradient information while making clip_grad_norm_
    always see finite values.  If the loss itself is NaN, all grads are
    zeroed (the forward pass produced garbage, so no gradient is usable).
    """

    def training_step(self, model, inputs, **kwargs):
        loss = super().training_step(model, inputs, **kwargs)

        if not torch.isfinite(loss):
            # Loss is NaN/Inf — zero everything, skip this batch entirely
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            return torch.tensor(0.0, device=loss.device, requires_grad=False)

        # Sanitise individual gradients: replace NaN/Inf with 0 in-place
        # so clip_grad_norm_ always sees finite values
        for p in model.parameters():
            if p.grad is not None:
                bad_mask = ~torch.isfinite(p.grad)
                if bad_mask.any():
                    p.grad[bad_mask] = 0.0
        return loss


# ── Gradient-Aware Polynomial Fine-Tuning ────────────────────────────────────


def calibrate_grad_norm(
    model: nn.Module,
    train_dataset,
    batch_size: int = 16,
    num_batches: int = 20,
    percentile: float = 90.0,
) -> float:
    """Measure actual gradient norms and return a calibrated clipping threshold.

    Polynomial activations amplify gradients proportionally to model depth.
    Standard gradient clipping (max_norm=1.0), designed for smooth activations
    like GELU and Softmax, suppresses >99% of gradient signal in polynomial
    models.  This calibration function runs a few forward-backward passes to
    measure the *actual* gradient norm distribution and returns a threshold at
    the given percentile — preserving the learning signal while still guarding
    against true gradient explosions.

    Parameters
    ----------
    model : nn.Module
        The model with polynomial activations already replaced.
    train_dataset : Dataset
        HuggingFace-formatted training dataset.
    batch_size : int
        Batch size for calibration (does NOT consume GPU memory for optimizer).
    num_batches : int
        Number of batches to sample.
    percentile : float
        The percentile of observed norms to use as the clipping threshold.
        90th percentile keeps 90% of gradients unclipped.

    Returns
    -------
    float
        Recommended max_grad_norm value.
    """
    device = next(model.parameters()).device
    model.train()

    sampler = RandomSampler(train_dataset, num_samples=batch_size * num_batches)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
    )

    grad_norms: List[float] = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if not torch.isfinite(loss):
            continue
        loss.backward()

        # Compute total gradient norm (same as clip_grad_norm_ uses)
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf")
        ).item()
        if np.isfinite(total_norm):
            grad_norms.append(total_norm)

    if not grad_norms:
        print("  [calibrate] WARNING: no finite gradients observed, using default 1.0")
        return 1.0

    calibrated = float(np.percentile(grad_norms, percentile))
    median = float(np.median(grad_norms))
    print(f"  [calibrate] Gradient norms over {len(grad_norms)} batches:")
    print(
        f"    median={median:.1f}, p{percentile:.0f}={calibrated:.1f}, "
        f"max={max(grad_norms):.1f}"
    )
    print(f"  [calibrate] Setting max_grad_norm = {calibrated:.1f}")

    # Reset model gradients
    model.zero_grad()
    return calibrated


# ── Knowledge Distillation for Polynomial Fine-Tuning ────────────────────────


class DistillationTrainer(NaNSafeTrainer):
    """NaN-safe trainer with knowledge distillation from a teacher model.

    Inherits gradient sanitisation from NaNSafeTrainer and adds:
    - KD loss: alpha * CE + (1-alpha) * T^2 * KL(student/T, teacher/T)
    - Optional gradient norm scheduling (linear decay over training)

    Parameters
    ----------
    teacher_model : nn.Module
        Frozen standard-activation model (the baseline).
    alpha : float
        Weight for the task loss (CE). Default: 0.5.
    temperature : float
        Softmax temperature for KD. Higher = softer targets. Default: 4.0.
    initial_max_grad_norm : float or None
        Starting max_grad_norm for gradient clipping schedule.
    final_max_grad_norm : float or None
        Ending max_grad_norm.  If both are set, the norm decays linearly.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        alpha: float = 0.5,
        temperature: float = 4.0,
        initial_max_grad_norm: Optional[float] = None,
        final_max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.temperature = temperature
        self.initial_max_grad_norm = initial_max_grad_norm
        self.final_max_grad_norm = final_max_grad_norm

    def training_step(self, model, inputs, **kwargs):
        loss = super().training_step(model, inputs, **kwargs)

        # Apply scheduled gradient norm clipping
        if (
            self.initial_max_grad_norm is not None
            and self.final_max_grad_norm is not None
        ):
            total_steps = self.state.max_steps
            current_step = self.state.global_step
            progress = min(current_step / max(total_steps, 1), 1.0)
            scheduled_norm = self.initial_max_grad_norm + progress * (
                self.final_max_grad_norm - self.initial_max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), scheduled_norm)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward (includes labels → computes CE loss internally)
        outputs = model(**inputs)
        task_loss = outputs.loss

        # During eval, skip teacher — eval accuracy only needs student outputs
        if not model.training:
            return (task_loss, outputs) if return_outputs else task_loss

        student_logits = outputs.logits

        # Teacher forward — move teacher to student device for training speed
        with torch.no_grad():
            student_device = student_logits.device
            self.teacher.to(student_device)
            teacher_inputs = {
                k: v.to(student_device) for k, v in inputs.items() if k != "labels"
            }
            teacher_logits = self.teacher(**teacher_inputs).logits

        # For regression heads (num_labels=1) KL is undefined; use MSE on
        # raw logits between student and teacher instead.
        if student_logits.shape[-1] == 1:
            kd_loss = F.mse_loss(student_logits.float(), teacher_logits.float())
        else:
            # KL divergence between temperature-smoothed distributions.
            T = self.temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

        loss = self.alpha * task_loss + (1.0 - self.alpha) * kd_loss
        return (loss, outputs) if return_outputs else loss


class AttentionDistillationTrainer(NaNSafeTrainer):
    """Trainer for Stage 2 (Softmax replacement) with attention-level KD.

    Combines three loss terms:
      L = alpha * L_CE
        + beta  * L_attn   (per-layer attention KL divergence)
        + gamma * L_hid    (per-layer hidden-state MSE)

    The attention KL loss gives each polynomial Softmax layer a DIRECT
    gradient signal: "your attention distribution differs from the teacher's
    real Softmax by this much."  Without it, the polynomial only receives
    diluted backprop through the entire network, which is too weak for the
    larger models (Mini/Small/Base) where Stage 2 accuracy drops 5%+.

    Literature basis: TinyBERT (Jiao et al., 2020), MiniLM (Wang et al., 2020).

    Parameters
    ----------
    teacher_model : nn.Module
        Stage 1 model with original Softmax (frozen).
    alpha : float
        Weight for task CE loss.
    beta : float
        Weight for attention distribution matching loss (KL divergence).
    gamma : float
        Weight for hidden state matching loss (MSE).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_training = model.training

        # --- Student forward ---
        # Only request attention/hidden outputs during training (for KD loss).
        # During eval, extra outputs cause inhomogeneous prediction arrays.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            output_attentions=is_training,
            output_hidden_states=is_training,
        )
        task_loss = outputs.loss

        # During eval, skip teacher
        if not is_training:
            return (task_loss, outputs) if return_outputs else task_loss

        student_attns = outputs.attentions  # tuple of [B, H, S, S]
        student_hiddens = outputs.hidden_states  # tuple of [B, S, D]

        # --- Teacher forward with attention outputs ---
        with torch.no_grad():
            device = outputs.logits.device
            self.teacher.to(device)
            teacher_out = self.teacher(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                output_attentions=True,
                output_hidden_states=True,
            )
        teacher_attns = teacher_out.attentions
        teacher_hiddens = teacher_out.hidden_states

        # --- Attention distribution matching (per-layer KL divergence) ---
        attn_loss = torch.tensor(0.0, device=device)
        num_layers = min(len(student_attns), len(teacher_attns))
        for l in range(num_layers):
            # student/teacher: [B, H, S, S] — already probability distributions
            s_attn = student_attns[l].float().clamp(min=1e-8)
            t_attn = teacher_attns[l].float().clamp(min=1e-8)
            # KL(teacher || student) per position, averaged over batch/heads/rows
            attn_loss = attn_loss + F.kl_div(
                s_attn.log(), t_attn, reduction="batchmean"
            )
        attn_loss = attn_loss / max(num_layers, 1)

        # --- Hidden state matching (per-layer MSE) ---
        hid_loss = torch.tensor(0.0, device=device)
        # hidden_states includes embedding output at index 0
        num_h = min(len(student_hiddens), len(teacher_hiddens))
        for l in range(num_h):
            hid_loss = hid_loss + F.mse_loss(
                student_hiddens[l].float(), teacher_hiddens[l].float()
            )
        hid_loss = hid_loss / max(num_h, 1)

        # --- Combined loss ---
        loss = self.alpha * task_loss + self.beta * attn_loss + self.gamma * hid_loss

        # Log individual loss components periodically
        if hasattr(self, "state") and self.state.global_step % 100 == 0:
            print(
                f"  [AttnKD step {self.state.global_step}] "
                f"CE={task_loss.item():.4f} "
                f"AttnKL={attn_loss.item():.4f} "
                f"HidMSE={hid_loss.item():.6f} "
                f"total={loss.item():.4f}"
            )

        return (loss, outputs) if return_outputs else loss


def detect_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        try:
            x = torch.randn(2, 2, device="cuda")
            _ = x @ x
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except RuntimeError:
            pass
    print("  Using CPU (GPU unavailable or incompatible)")
    return torch.device("cpu")


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy for classification evaluation (legacy SST-2 default)."""
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {"accuracy": float((preds == eval_pred.label_ids).mean())}


def _binary_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec}


def compute_metrics_for_task(
    task: TaskConfig,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """Return a `compute_metrics` callable matching the task's metric set.

    Always reports `task.metric` as a top-level key so HuggingFace
    Trainer can use `metric_for_best_model = f"eval_{task.metric}"`.
    """
    if task.is_regression:
        # Lazy import: scipy is heavy and only required for STS-B.
        from scipy.stats import pearsonr, spearmanr

        def _metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            preds = eval_pred.predictions.squeeze().astype(np.float64)
            labels = eval_pred.label_ids.astype(np.float64)
            pearson = float(pearsonr(preds, labels).statistic)
            spearman = float(spearmanr(preds, labels).statistic)
            return {"pearson": pearson, "spearmanr": spearman}

        return _metrics

    def _metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        out = {"accuracy": float((preds == labels).mean())}
        if "f1" in task.eval_metrics:
            out.update(_binary_f1(preds, labels))
        return out

    return _metrics


def load_glue_dataset(
    task: TaskConfig,
    tokenizer,
    max_length: int = 128,
    splits: Optional[Tuple[str, ...]] = None,
):
    """Load and tokenize any GLUE task; returns (train, eval_dict).

    Returns a tuple (train_dataset, eval_datasets) where eval_datasets is
    a dict mapping split label -> dataset.  For most tasks the dict has a
    single 'validation' key; for MNLI it has 'matched' and 'mismatched'.
    The first entry is the one used for best-model selection.
    """
    dataset = load_dataset("glue", task.name)
    fields = task.text_fields

    def tokenize_fn(examples):
        if len(fields) == 1:
            return tokenizer(
                examples[fields[0]],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            examples[fields[0]],
            examples[fields[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(task.label_column, "labels")

    # HuggingFace stores STS-B labels as float32 already; for classification
    # tasks they are int64.  No explicit cast needed for either case.
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    use_splits = splits if splits is not None else task.eval_splits
    eval_datasets: Dict[str, Any] = {}
    for sp in use_splits:
        # MNLI: keep human-readable short labels ("matched", "mismatched").
        short = sp.replace("validation_", "") if sp.startswith("validation_") else sp
        eval_datasets[short] = tokenized[sp]

    return tokenized["train"], eval_datasets


def load_sst2_dataset(tokenizer, max_length: int = 128):
    """Load and tokenize the SST-2 dataset.

    Returns
    -------
    tuple
        (train_dataset, eval_dataset)
    """
    dataset = load_dataset("glue", "sst2")

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized["train"], tokenized["validation"]


def train_and_eval(
    model: nn.Module,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    label: str,
    use_fp16: bool = True,
    max_grad_norm: Optional[float] = None,
    compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict[str, float]]] = None,
    metric_for_best_model: str = "eval_accuracy",
) -> Dict[str, Any]:
    """Train and evaluate a model, return results dict."""
    device = detect_device()

    extra_kwargs = {}
    if max_grad_norm is not None:
        extra_kwargs["max_grad_norm"] = max_grad_norm

    metric_fn = (
        compute_metrics_fn if compute_metrics_fn is not None else compute_metrics
    )
    primary_key = (
        metric_for_best_model[len("eval_") :]
        if metric_for_best_model.startswith("eval_")
        else metric_for_best_model
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        load_best_model_at_end=False,
        report_to="none",
        disable_tqdm=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
        dataloader_num_workers=4,
        **extra_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
    )

    print(f"  Training '{label}' for {epochs} epochs (bs={batch_size}, lr={lr})...")
    trainer.train()
    results = trainer.evaluate()
    acc = results.get(f"eval_{primary_key}", float("nan"))
    print(f"  {label} {primary_key}: {acc:.4f}")

    # Make all tensors contiguous before saving
    for p in model.parameters():
        p.data = p.data.contiguous()
    try:
        trainer.save_model(os.path.join(output_dir, "best_model"))
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    return {
        "label": label,
        "accuracy": acc,
        "eval_loss": results["eval_loss"],
        "epochs": epochs,
    }


def distill_and_eval(
    teacher_model: nn.Module,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    label: str,
    use_fp16: bool = False,
    max_grad_norm: Optional[float] = None,
    initial_max_grad_norm: Optional[float] = None,
    final_max_grad_norm: Optional[float] = None,
    alpha: float = 0.5,
    temperature: float = 4.0,
    seed: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    gradient_accumulation_steps: int = 1,
    compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict[str, float]]] = None,
    metric_for_best_model: str = "eval_accuracy",
) -> Dict[str, Any]:
    """Fine-tune a polynomial model using knowledge distillation.

    Parameters
    ----------
    student_model : nn.Module
        The polynomial-activation model to train.
    teacher_model : nn.Module
        The standard-activation baseline model (frozen, on same device).
    alpha : float
        Blend factor: alpha * CE + (1-alpha) * KD_loss.
    temperature : float
        Softmax temperature for distillation.
    """
    device = detect_device()

    extra_kwargs = {}
    if max_grad_norm is not None:
        extra_kwargs["max_grad_norm"] = max_grad_norm
    if seed is not None:
        extra_kwargs["seed"] = seed
        extra_kwargs["data_seed"] = seed

    metric_fn = (
        compute_metrics_fn if compute_metrics_fn is not None else compute_metrics
    )
    primary_key = (
        metric_for_best_model[len("eval_") :]
        if metric_for_best_model.startswith("eval_")
        else metric_for_best_model
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",
        disable_tqdm=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
        dataloader_num_workers=4,
        **extra_kwargs,
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=alpha,
        temperature=temperature,
        initial_max_grad_norm=initial_max_grad_norm,
        final_max_grad_norm=final_max_grad_norm,
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(
        f"  Training '{label}' for {epochs} epochs "
        f"(KD: alpha={alpha}, T={temperature}, bs={batch_size}, lr={lr})..."
    )
    if resume_from_checkpoint:
        print(f"  Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"  {label} accuracy: {acc:.4f} ({acc:.2%})")

    # Make tensors contiguous before saving
    for p in student_model.parameters():
        p.data = p.data.contiguous()
    try:
        trainer.save_model(os.path.join(output_dir, "best_model"))
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    return {
        "label": label,
        "accuracy": acc,
        "eval_loss": results["eval_loss"],
        "epochs": epochs,
    }


def attn_distill_and_eval(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    label: str,
    use_fp16: bool = False,
    max_grad_norm: Optional[float] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    seed: Optional[int] = None,
    lr_scheduler_type: str = "cosine",
    compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict[str, float]]] = None,
    metric_for_best_model: str = "eval_accuracy",
) -> Dict[str, Any]:
    """Fine-tune Stage 2 (Softmax) with attention-level knowledge distillation.

    Uses AttentionDistillationTrainer which combines:
      L = alpha * L_CE + beta * L_attn_KL + gamma * L_hidden_MSE

    Parameters
    ----------
    alpha : float
        Weight for task CE loss.
    beta : float
        Weight for per-layer attention distribution KL divergence.
    gamma : float
        Weight for per-layer hidden state MSE.
    """
    device = detect_device()

    extra_kwargs = {}
    if max_grad_norm is not None:
        extra_kwargs["max_grad_norm"] = max_grad_norm
    if seed is not None:
        extra_kwargs["seed"] = seed
        extra_kwargs["data_seed"] = seed

    metric_fn = (
        compute_metrics_fn if compute_metrics_fn is not None else compute_metrics
    )
    primary_key = (
        metric_for_best_model[len("eval_") :]
        if metric_for_best_model.startswith("eval_")
        else metric_for_best_model
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",
        disable_tqdm=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        no_cuda=device.type == "cpu",
        dataloader_num_workers=4,
        **extra_kwargs,
    )

    trainer = AttentionDistillationTrainer(
        teacher_model=teacher_model,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(
        f"  Training '{label}' for {epochs} epochs "
        f"(AttnKD: alpha={alpha}, beta={beta}, gamma={gamma}, "
        f"bs={batch_size}, lr={lr})..."
    )
    trainer.train()
    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"  {label} accuracy: {acc:.4f} ({acc:.2%})")

    # Make tensors contiguous before saving
    for p in student_model.parameters():
        p.data = p.data.contiguous()
    try:
        trainer.save_model(os.path.join(output_dir, "best_model"))
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")

    return {
        "label": label,
        "accuracy": acc,
        "eval_loss": results["eval_loss"],
        "epochs": epochs,
    }
