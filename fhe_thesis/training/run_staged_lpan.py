"""CLI for Stage-1 to Stage-3 LPAN training.

This runner produces the polynomial teacher chain required by
``run_synth_lpan.py``:

- baseline fine-tuning
- Stage 1: GELU -> learnable polynomial
- Stage 2: Softmax -> learnable polynomial with attention KD
- Stage 3: LayerNorm -> learnable polynomial with KD
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from fhe_thesis.config import MODEL_REGISTRY, SYNTHESIZER_LPAN_DIR, ensure_dirs
from fhe_thesis.models import compute_poly_coefficients, profile_model, replace_activations
from fhe_thesis.models.replacement import build_poly_config_from_state_dict
from fhe_thesis.tasks import GLUE_TASKS, get_task
from fhe_thesis.training.checkpoints import (
    load_checkpoint_state_dict,
    mark_stage_done,
    stage_done,
)
from fhe_thesis.training.trainer import (
    attn_distill_and_eval,
    compute_metrics_for_task,
    distill_and_eval,
    load_glue_dataset,
    train_and_eval,
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
    parser = argparse.ArgumentParser(description="Run Stage-1 to Stage-3 LPAN training")
    parser.add_argument("--model", required=True, help="Model key or alias (e.g. base, bert-base)")
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TASKS), help="GLUE task")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "baseline", "1", "2", "3"],
        help="Run all stages or stop after a specific stage",
    )
    parser.add_argument("--output-dir", default=None, help="Output root (default: results/synthesizer_lpan/<task>/<model>)")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization max length")
    parser.add_argument("--profile-samples", type=int, default=1000, help="Train samples used to profile activation ranges")
    parser.add_argument("--poly-degree", type=int, default=8, help="Base polynomial degree before op/depth-specific adjustment")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Training precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--llrd", action="store_true", help="Enable layer-wise learning-rate decay")
    parser.add_argument("--llrd-decay", type=float, default=0.95, help="LLRD decay factor")
    parser.add_argument("--force", action="store_true", help="Re-run stages even if .done markers exist")
    parser.add_argument("--baseline-epochs", type=int, default=None, help="Override baseline epochs")
    parser.add_argument("--stage1-epochs", type=int, default=3, help="Stage 1 epochs")
    parser.add_argument("--stage2-epochs", type=int, default=3, help="Stage 2 epochs")
    parser.add_argument("--stage3-epochs", type=int, default=5, help="Stage 3 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--baseline-lr", type=float, default=None, help="Baseline learning rate")
    parser.add_argument("--stage1-lr", type=float, default=None, help="Stage 1 learning rate")
    parser.add_argument("--stage2-lr", type=float, default=None, help="Stage 2 learning rate")
    parser.add_argument("--stage3-lr", type=float, default=None, help="Stage 3 learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Optional gradient clipping threshold")
    return parser.parse_args()


def _normalize_model_key(model_key: str) -> str:
    key = model_key.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{model_key}'. Available keys: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_ALIASES[key]


def _serialize_poly_cfg(poly_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in poly_cfg.items():
        out[key] = {
            "cheb_coeffs": np.asarray(value["cheb_coeffs"]).tolist(),
            "interval": [float(value["interval"][0]), float(value["interval"][1])],
            "degree": int(value["degree"]),
        }
    return out


def _save_interval_overrides(model: torch.nn.Module, path: Path) -> None:
    intervals: Dict[str, list[float]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "a") and hasattr(module, "b"):
            intervals[name] = [float(module.a), float(module.b)]
    path.write_text(json.dumps(intervals, indent=2) + "\n", encoding="utf-8")


def _load_interval_overrides(path: Path) -> Optional[Dict[str, Sequence[float]]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): [float(v[0]), float(v[1])] for k, v in data.items()}


def _save_stage_artifacts(model: torch.nn.Module, tokenizer, stage_dir: Path) -> None:
    best_model_dir = stage_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(best_model_dir)
    _save_interval_overrides(model, best_model_dir / "intervals.json")
    mark_stage_done(stage_dir)


def _build_stage_model(checkpoint_dir: Path) -> torch.nn.Module:
    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    interval_overrides = _load_interval_overrides(checkpoint_dir / "intervals.json")
    poly_cfg = build_poly_config_from_state_dict(state_dict, interval_overrides)
    config = AutoConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_config(
        config,
        attn_implementation="eager",
    )
    if poly_cfg:
        replace_activations(
            model,
            poly_cfg,
            hidden_size=config.hidden_size,
            learnable=True,
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def _build_baseline_model(model_name: str, task_name: str, num_labels: int, problem_type: str):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        finetuning_task=task_name,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        attn_implementation="eager",
    )


def _should_stop(stage_arg: str, current_stage: str) -> bool:
    return stage_arg == current_stage


def main() -> int:
    args = _parse_args()
    ensure_dirs()

    model_key = _normalize_model_key(args.model)
    model_cfg = MODEL_REGISTRY[model_key]
    task = get_task(args.task)
    batch_size = args.batch_size or model_cfg["batch_size"]
    output_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else SYNTHESIZER_LPAN_DIR / task.name / model_key
    )
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_root / "baseline"
    stage1_dir = output_root / "staged_lpan_s1_gelu"
    stage2_dir = output_root / "staged_lpan_s2_softmax"
    stage3_dir = output_root / "staged_lpan_s3_ln_kd"

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    train_dataset, eval_datasets = load_glue_dataset(task, tokenizer, max_length=args.max_length)
    eval_dataset = next(iter(eval_datasets.values()))
    metric_fn = compute_metrics_for_task(task)

    baseline_ckpt = baseline_dir / "best_model"
    if not stage_done(baseline_dir) or args.force:
        baseline_model = _build_baseline_model(
            model_cfg["name"],
            task.name,
            task.num_labels,
            task.problem_type,
        )
        train_and_eval(
            model=baseline_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(baseline_dir),
            epochs=args.baseline_epochs or task.epochs,
            batch_size=batch_size,
            lr=args.baseline_lr or model_cfg["lr"],
            label=f"baseline_{task.name}_{model_key}",
            precision=args.precision,
            llrd=args.llrd,
            llrd_decay=args.llrd_decay,
            max_grad_norm=args.max_grad_norm,
            compute_metrics_fn=metric_fn,
            metric_for_best_model=task.metric_for_best_model,
        )
        _save_stage_artifacts(baseline_model, tokenizer, baseline_dir)
    if _should_stop(args.stage, "baseline"):
        return 0

    poly_cfg_path = output_root / "poly_coeffs.json"
    if poly_cfg_path.exists() and not args.force:
        with open(poly_cfg_path, "r", encoding="utf-8") as handle:
            raw_poly_cfg = json.load(handle)
        poly_cfg = {
            key: {
                "cheb_coeffs": np.asarray(value["cheb_coeffs"], dtype=np.float32),
                "interval": (float(value["interval"][0]), float(value["interval"][1])),
                "degree": int(value["degree"]),
            }
            for key, value in raw_poly_cfg.items()
        }
    else:
        baseline_model = _build_stage_model(baseline_ckpt)
        profile_data = profile_model(
            model_name=model_cfg["name"],
            num_layers=baseline_model.config.num_hidden_layers,
            num_samples=args.profile_samples,
            split="train",
            model_obj=baseline_model,
            task_name=task.name,
            max_length=args.max_length,
        )
        poly_cfg = compute_poly_coefficients(
            profile_data,
            num_layers=baseline_model.config.num_hidden_layers,
            degree=args.poly_degree,
        )
        poly_cfg_path.write_text(
            json.dumps(_serialize_poly_cfg(poly_cfg), indent=2) + "\n",
            encoding="utf-8",
        )

    if not stage_done(stage1_dir) or args.force:
        stage1_model = _build_stage_model(baseline_ckpt)
        replace_activations(
            stage1_model,
            poly_cfg,
            hidden_size=stage1_model.config.hidden_size,
            learnable=True,
            replace_types=["GELU"],
        )
        train_and_eval(
            model=stage1_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(stage1_dir),
            epochs=args.stage1_epochs,
            batch_size=batch_size,
            lr=args.stage1_lr or model_cfg["lr"],
            label=f"stage1_gelu_{task.name}_{model_key}",
            precision=args.precision,
            llrd=args.llrd,
            llrd_decay=args.llrd_decay,
            max_grad_norm=args.max_grad_norm,
            compute_metrics_fn=metric_fn,
            metric_for_best_model=task.metric_for_best_model,
        )
        _save_stage_artifacts(stage1_model, tokenizer, stage1_dir)
    if _should_stop(args.stage, "1"):
        return 0

    if not stage_done(stage2_dir) or args.force:
        stage1_teacher = _build_stage_model(stage1_dir / "best_model")
        stage2_student = copy.deepcopy(stage1_teacher)
        replace_activations(
            stage2_student,
            poly_cfg,
            hidden_size=stage2_student.config.hidden_size,
            learnable=True,
            replace_types=["Softmax"],
        )
        attn_distill_and_eval(
            student_model=stage2_student,
            teacher_model=stage1_teacher,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(stage2_dir),
            epochs=args.stage2_epochs,
            batch_size=batch_size,
            lr=args.stage2_lr or model_cfg["lr"],
            label=f"stage2_softmax_{task.name}_{model_key}",
            precision=args.precision,
            llrd=args.llrd,
            llrd_decay=args.llrd_decay,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            compute_metrics_fn=metric_fn,
            metric_for_best_model=task.metric_for_best_model,
        )
        _save_stage_artifacts(stage2_student, tokenizer, stage2_dir)
    if _should_stop(args.stage, "2"):
        return 0

    if not stage_done(stage3_dir) or args.force:
        stage2_teacher = _build_stage_model(stage2_dir / "best_model")
        stage3_student = copy.deepcopy(stage2_teacher)
        replace_activations(
            stage3_student,
            poly_cfg,
            hidden_size=stage3_student.config.hidden_size,
            learnable=True,
            replace_types=["LN"],
        )
        distill_and_eval(
            student_model=stage3_student,
            teacher_model=stage2_teacher,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(stage3_dir),
            epochs=args.stage3_epochs,
            batch_size=batch_size,
            lr=args.stage3_lr or (1e-5 if model_key == "base" else model_cfg["lr"]),
            label=f"stage3_ln_{task.name}_{model_key}",
            precision=args.precision,
            llrd=args.llrd,
            llrd_decay=args.llrd_decay,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            compute_metrics_fn=metric_fn,
            metric_for_best_model=task.metric_for_best_model,
        )
        _save_stage_artifacts(stage3_student, tokenizer, stage3_dir)

    summary = {
        "status": "ok",
        "task": task.name,
        "model": model_key,
        "output_root": str(output_root),
        "profile_samples": args.profile_samples,
        "poly_degree": args.poly_degree,
        "stages": {
            "baseline": str(baseline_dir / "best_model"),
            "stage1": str(stage1_dir / "best_model"),
            "stage2": str(stage2_dir / "best_model"),
            "stage3": str(stage3_dir / "best_model"),
        },
    }
    (output_root / "staged_lpan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())