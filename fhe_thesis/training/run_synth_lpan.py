"""CLI for Stage-4 Synthesizer-LPAN training.

Current scope: Stage 4 only.
Loads a Stage-3 LPAN checkpoint, swaps encoder self-attention blocks to
SynthesizerAttention, distills against the frozen Stage-3 teacher, and saves a
production-ready checkpoint tree under ``results/synthesizer_lpan/<task>/<model>``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import default_data_collator

from fhe_thesis.config import MODEL_REGISTRY, SYNTHESIZER_LPAN_DIR, ensure_dirs
from fhe_thesis.models import replace_attention_with_synthesizer
from fhe_thesis.models.replacement import build_poly_config_from_state_dict, replace_activations
from fhe_thesis.tasks import GLUE_TASKS, get_task
from fhe_thesis.training.checkpoints import find_lpan_checkpoint, load_checkpoint_state_dict
from fhe_thesis.training.trainer import (
    compute_metrics_for_task,
    detect_device,
    load_glue_dataset,
    synth_attn_distill_and_eval,
)


MODEL_ALIASES = {
    "tiny": "tiny",
    "bert-tiny": "tiny",
    "mini": "mini",
    "bert-mini": "mini",
    "small": "small",
    "bert-small": "small",
    "base": "base",
    "bert-base": "base",
    "bert-base-uncased": "base",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert",
    "distilbert-base-uncased": "distilbert",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-4 Synthesizer-LPAN training")
    parser.add_argument("--model", required=True, help="Model key or alias (e.g. base, bert-base)")
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TASKS), help="GLUE task")
    parser.add_argument("--stage", type=int, default=4, help="Training stage to run (currently only 4)")
    parser.add_argument("--lpan-checkpoint", default=None, help="Override Stage-3 checkpoint directory")
    parser.add_argument("--intervals-json", default=None, help="Optional interval metadata JSON for legacy Stage-3 checkpoints")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/synthesizer_lpan/<task>/<model>)")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization and Synthesizer sequence length")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Training precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--llrd", action="store_true", help="Enable layer-wise learning-rate decay")
    parser.add_argument("--llrd-decay", type=float, default=0.95, help="LLRD decay factor")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--distill-weight-start", type=float, default=1.0, help="Initial attention-KL weight")
    parser.add_argument("--distill-weight-end", type=float, default=0.0, help="Final attention-KL weight")
    parser.add_argument("--init-batches", type=int, default=16, help="Teacher batches to average when seeding Synthesizer patterns")
    return parser.parse_args()


def _normalize_model_key(model_key: str) -> str:
    key = model_key.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{model_key}'. Available keys: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_ALIASES[key]


def _load_interval_overrides(path: Optional[str]) -> Optional[Dict[str, Sequence[float]]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): [float(v[0]), float(v[1])] for k, v in data.items()}


def _save_interval_overrides(model: torch.nn.Module, path: Path) -> None:
    intervals: Dict[str, list[float]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "a") and hasattr(module, "b"):
            intervals[name] = [float(module.a), float(module.b)]
    path.write_text(json.dumps(intervals, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def _estimate_attention_patterns(
    teacher_model: torch.nn.Module,
    train_dataset,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> Optional[Dict[int, torch.Tensor]]:
    if max_batches <= 0:
        return None

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    sums: Dict[int, torch.Tensor] = {}
    total_samples = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask"}
        }
        outputs = teacher_model(**batch, output_attentions=True)
        batch_size_now = int(outputs.attentions[0].shape[0])
        total_samples += batch_size_now
        for layer_idx, attn in enumerate(outputs.attentions):
            if attn is None:
                continue
            layer_sum = attn.detach().sum(dim=0)
            if layer_idx not in sums:
                sums[layer_idx] = layer_sum
            else:
                sums[layer_idx] = sums[layer_idx] + layer_sum

    teacher_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if total_samples == 0:
        return None
    return {layer_idx: (tensor / total_samples).cpu() for layer_idx, tensor in sums.items()}


def _build_stage3_model(
    checkpoint_dir: Path,
    interval_overrides: Optional[Dict[str, Sequence[float]]],
) -> tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    poly_cfg = build_poly_config_from_state_dict(state_dict, interval_overrides)
    config = AutoConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_config(
        config,
        attn_implementation="eager",
    )
    replace_activations(
        model,
        poly_cfg,
        hidden_size=config.hidden_size,
        learnable=True,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, state_dict


def main() -> int:
    args = _parse_args()
    if args.stage != 4:
        raise ValueError("Only --stage 4 is implemented in this CLI")

    ensure_dirs()
    model_key = _normalize_model_key(args.model)
    task = get_task(args.task)
    model_cfg = MODEL_REGISTRY[model_key]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else SYNTHESIZER_LPAN_DIR / task.name / model_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_ckpt = find_lpan_checkpoint(model_key, task.name, override=args.lpan_checkpoint)
    interval_path = Path(args.intervals_json) if args.intervals_json else stage3_ckpt / "intervals.json"
    interval_overrides = _load_interval_overrides(str(interval_path)) if interval_path.exists() else None

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    train_dataset, eval_datasets = load_glue_dataset(task, tokenizer, max_length=args.max_length)
    eval_dataset = next(iter(eval_datasets.values()))

    teacher_model, _ = _build_stage3_model(stage3_ckpt, interval_overrides)

    init_patterns = None
    if args.init_batches != 0:
        device = detect_device()
        init_patterns = _estimate_attention_patterns(
            teacher_model,
            train_dataset,
            batch_size=args.batch_size or model_cfg["batch_size"],
            max_batches=args.init_batches,
            device=device,
        )

    student_model = copy.deepcopy(teacher_model)
    replace_attention_with_synthesizer(
        student_model,
        max_seq_len=args.max_length,
        init_patterns=init_patterns,
    )

    result = synth_attn_distill_and_eval(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir),
        epochs=args.epochs or task.epochs,
        batch_size=args.batch_size or model_cfg["batch_size"],
        lr=args.lr or model_cfg["lr"],
        label=f"stage4_synth_{task.name}_{model_key}",
        precision=args.precision,
        llrd=args.llrd,
        llrd_decay=args.llrd_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        distill_weight_start=args.distill_weight_start,
        distill_weight_end=args.distill_weight_end,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compute_metrics_fn=compute_metrics_for_task(task),
        metric_for_best_model=task.metric_for_best_model,
    )

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(best_model_dir)
    _save_interval_overrides(student_model, best_model_dir / "intervals.json")

    summary = {
        "status": "ok",
        "stage": 4,
        "task": task.name,
        "model": model_key,
        "source_stage3": str(stage3_ckpt),
        "output": str(best_model_dir),
        "max_length": args.max_length,
        "epochs": args.epochs or task.epochs,
        "batch_size": args.batch_size or model_cfg["batch_size"],
        "lr": args.lr or model_cfg["lr"],
        "precision": args.precision,
        "distill_weight_start": args.distill_weight_start,
        "distill_weight_end": args.distill_weight_end,
        "init_batches": args.init_batches,
        "final_metrics": result,
    }
    (output_dir / "stage4_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())