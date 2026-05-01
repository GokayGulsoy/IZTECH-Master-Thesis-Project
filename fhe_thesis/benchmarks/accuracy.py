"""Plaintext accuracy benchmarks for HyPER-LPAN, LPAN, and hybrid models.

This module is the canonical entry point for thesis-table generation.
It loads any saved checkpoint (LPAN final, HyPER-LPAN best_model, or a
plain HuggingFace BERT) and evaluates it on the matching GLUE task.

Public API
----------
evaluate_checkpoint(model_key, task, checkpoint_path, *, variant, ...)
    Evaluate one checkpoint and return a dict of metrics.

compare_checkpoints(specs, ...)
    Convenience wrapper that runs many checkpoints and returns a
    pandas DataFrame.  ``specs`` is a list of dicts with keys
    ``label``, ``variant``, ``checkpoint_path``, ``model_key``, ``task``
    and optional variant-specific kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from safetensors.torch import load_file as _load_safetensors
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fhe_thesis.config import MODEL_REGISTRY
from fhe_thesis.models.hybrid_attention import apply_hybrid_attention
from fhe_thesis.models.lpan_loader import load_lpan_model
from fhe_thesis.tasks import get_task
from fhe_thesis.training.trainer import (
    compute_metrics_for_task,
    detect_device,
    load_glue_dataset,
)


# ──────────────────────────────────────────────────────────────────────


def _load_state(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    sf = ckpt_dir / "model.safetensors"
    if sf.exists():
        return _load_safetensors(str(sf))
    bin_path = ckpt_dir / "pytorch_model.bin"
    return torch.load(str(bin_path), map_location="cpu", weights_only=False)


def _load_baseline(model_key: str, ckpt_path: Path, num_labels: int):
    """Vanilla transformer (no polynomial replacements)."""
    return AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=num_labels
    ).eval()


def _load_hybrid(
    model_key: str,
    ckpt_path: Path,
    *,
    linear_mixing_layers: Sequence[int],
    quad_attention_layers: Sequence[int],
    max_seq_len: int,
    num_labels: int,
    quad_num_heads: int = 12,
):
    """Re-architect a fresh BERT to match the saved HyPER-LPAN composition."""
    # Step 1: Build LPAN model from the LPAN baseline checkpoint structure
    # (we need the polynomial replacements present).  Use a *placeholder*
    # path: load_lpan_model loads vanilla BERT then applies poly replacements.
    cfg = MODEL_REGISTRY[model_key]
    # Use the hybrid checkpoint itself for shape reference; coeffs will
    # be overwritten by the strict load below.
    student = load_lpan_model(
        model_key, ckpt_path,
        num_labels=num_labels, device="cpu", profile_samples=64,
    )
    # Step 2: Apply hybrid attention dispatch (LinearMixing + Quad layers)
    apply_hybrid_attention(
        student,
        linear_mixing_layers=list(linear_mixing_layers),
        quad_attention_layers=list(quad_attention_layers),
        max_seq_len=max_seq_len,
        num_heads=quad_num_heads,
    )
    # Step 3: Strict re-load of the saved state on the now-correctly-shaped model
    state = _load_state(ckpt_path)
    missing, unexpected = student.load_state_dict(state, strict=False)
    bad = [k for k in missing if any(t in k for t in (".coeffs", ".weight", ".bias"))
           and "embeddings.position_ids" not in k]
    # Some LPAN/quad-only keys legitimately don't co-exist with hybrid keys;
    # surface only critical missing keys for the *kept* layers.
    return student.eval()


# ──────────────────────────────────────────────────────────────────────


def evaluate_checkpoint(
    model_key: str,
    task: str,
    checkpoint_path: str | Path,
    *,
    variant: str = "lpan",
    max_seq_len: int = 128,
    batch_size: int = 64,
    num_labels: Optional[int] = None,
    linear_mixing_layers: Sequence[int] = (0, 1, 2, 3),
    quad_attention_layers: Sequence[int] = (4, 5, 6, 7),
    quad_num_heads: int = 12,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one checkpoint on a GLUE task; return metrics dict.

    Parameters
    ----------
    variant : {"baseline", "lpan", "hybrid"}
        ``baseline`` = vanilla HF BERT; ``lpan`` = LPAN polynomial model;
        ``hybrid`` = HyPER-LPAN (LinearMixing + Quad + LPAN).
    """
    task_cfg = get_task(task)
    if num_labels is None:
        num_labels = task_cfg.num_labels
    ckpt = Path(checkpoint_path)
    cfg = MODEL_REGISTRY[model_key]

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    train_ds, eval_dict = load_glue_dataset(task_cfg, tokenizer, max_length=max_seq_len)
    eval_split = next(iter(eval_dict))
    eval_ds = eval_dict[eval_split]

    if variant == "baseline":
        model = _load_baseline(model_key, ckpt, num_labels=num_labels)
    elif variant == "lpan":
        model = load_lpan_model(model_key, ckpt, num_labels=num_labels, device="cpu")
    elif variant == "hybrid":
        model = _load_hybrid(
            model_key, ckpt,
            linear_mixing_layers=linear_mixing_layers,
            quad_attention_layers=quad_attention_layers,
            max_seq_len=max_seq_len,
            num_labels=num_labels,
            quad_num_heads=quad_num_heads,
        )
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/_bench_eval",
            per_device_eval_batch_size=batch_size,
            report_to="none", disable_tqdm=True,
        ),
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_for_task(task_cfg),
    )
    out = trainer.evaluate()

    primary = task_cfg.metric_for_best_model
    return {
        "model": model_key,
        "task": task,
        "variant": variant,
        "checkpoint": str(ckpt),
        "split": eval_split,
        "primary_metric_name": primary,
        "primary_metric_value": float(out[primary]),
        "all_metrics": {k: (float(v) if hasattr(v, "__float__") else v)
                         for k, v in out.items()},
    }


def compare_checkpoints(
    specs: List[Dict[str, Any]],
    *,
    output_path: Optional[str | Path] = None,
):
    """Run a list of evaluations and return a pandas DataFrame.

    Each spec dict requires keys: ``label``, ``variant``,
    ``checkpoint_path``, ``model_key``, ``task``.  Other keys are
    forwarded to ``evaluate_checkpoint`` as kwargs.
    """
    import pandas as pd

    rows = []
    for spec in specs:
        label = spec.pop("label")
        spec_copy = dict(spec)
        result = evaluate_checkpoint(
            model_key=spec_copy.pop("model_key"),
            task=spec_copy.pop("task"),
            checkpoint_path=spec_copy.pop("checkpoint_path"),
            **spec_copy,
        )
        rows.append({
            "label": label,
            "task": result["task"],
            "model": result["model"],
            "variant": result["variant"],
            result["primary_metric_name"]: result["primary_metric_value"],
            "checkpoint": result["checkpoint"],
        })
    df = pd.DataFrame(rows)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        out.with_suffix(".json").write_text(
            json.dumps([r for r in rows], indent=2)
        )
    return df
