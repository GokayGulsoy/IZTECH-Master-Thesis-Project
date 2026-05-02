"""
Compute and cache FP32 vanilla BERT-* baselines per GLUE task.

Fine-tunes a clean pretrained BERT-base/large/etc. on the requested GLUE task
and writes the resulting metric to ``results/baselines/fp32_finetuned.json``.

The HyPER-LPAN pipeline reads this cache file at Stage A and stamps the
FP32 baseline into every per-run results.json so we can report the full
comparison: FP32 -> LPAN -> HyPER-LPAN.

Usage
-----
    python scripts/eval_fp32_baselines.py --model base --task mrpc
    python scripts/eval_fp32_baselines.py --model base --task all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fhe_thesis.tasks import GLUE_TASKS, get_task_config
from fhe_thesis.training.trainer import (
    compute_metrics_for_task,
    load_glue_dataset,
)

MODEL_HF_NAMES = {
    "tiny": "prajjwal1/bert-tiny",
    "mini": "prajjwal1/bert-mini",
    "small": "prajjwal1/bert-small",
    "medium": "prajjwal1/bert-medium",
    "base": "bert-base-uncased",
    "large": "bert-large-uncased",
}

CACHE_PATH = Path("results/baselines/fp32_finetuned.json")


def finetune_one(model_short: str, task: str, epochs: int, batch_size: int,
                 lr: float, max_seq_len: int, seed: int) -> dict:
    hf_name = MODEL_HF_NAMES[model_short]
    task_cfg = get_task_config(task)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    train_ds, eval_ds = load_glue_dataset(task_cfg, tokenizer, max_seq_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=task_cfg.num_labels,
    )

    out_dir = Path(f"results/baselines/_tmp_{model_short}_{task}")
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        seed=seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_for_task(task_cfg),
    )
    trainer.train()
    metrics = trainer.evaluate()

    primary = task_cfg.metric_for_best_model
    record = {
        "task": task,
        "model": model_short,
        "metric_name": task_cfg.metric,
        "metric_value": float(metrics[primary]),
        "all_metrics": {k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float))},
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }

    # Cleanup tmp output
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)

    return record


def update_cache(records: list[dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        table = json.loads(CACHE_PATH.read_text())
    else:
        table = {}
    for r in records:
        key = f"{r['model']}/{r['task']}"
        table[key] = r
    CACHE_PATH.write_text(json.dumps(table, indent=2))
    print(f"\nupdated -> {CACHE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_HF_NAMES))
    ap.add_argument("--task", required=True,
                    help="GLUE task name or 'all'")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task == "all":
        tasks = list(GLUE_TASKS)
    else:
        tasks = [args.task]

    records = []
    for t in tasks:
        print(f"\n=== fine-tuning vanilla {args.model} on {t} ===")
        r = finetune_one(args.model, t, args.epochs, args.batch_size,
                         args.lr, args.max_seq_len, args.seed)
        print(f"  -> {r['metric_name']} = {r['metric_value']:.4f}")
        records.append(r)

    update_cache(records)


if __name__ == "__main__":
    main()
