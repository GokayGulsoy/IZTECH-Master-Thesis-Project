"""GLUE task registry for multi-dataset LPAN evaluation.

Each entry describes how to load a GLUE task from HuggingFace `datasets`,
which fields to tokenize, the number of output labels, the problem type
(classification vs. regression), and which metric drives best-model
selection during training.

This module is consumed by:
  - fhe_thesis.training.trainer  (load_glue_dataset, compute_metrics_for_task)
  - experiments/05_multi_model_scaling.py  (baseline training per task)
  - run_staged_lpan.py            (3-stage LPAN per task)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single GLUE task."""

    name: str  # GLUE config name for load_dataset("glue", name)
    text_fields: Tuple[str, ...]  # 1 or 2 text columns to tokenize
    num_labels: int = 2  # 2/3 for classification, 1 for regression
    problem_type: str = "single_label_classification"  # or "regression"
    label_column: str = "label"
    metric: str = "accuracy"  # primary metric for best-model selection
    eval_metrics: Tuple[str, ...] = ("accuracy",)  # all metrics reported each epoch
    epochs: int = 3  # default baseline epoch count
    eval_splits: Tuple[str, ...] = ("validation",)  # MNLI uses matched + mismatched
    description: str = ""

    @property
    def is_regression(self) -> bool:
        return self.problem_type == "regression"

    @property
    def metric_for_best_model(self) -> str:
        """HuggingFace Trainer key (always prefixed with 'eval_')."""
        return f"eval_{self.metric}"


# ── GLUE task registry ───────────────────────────────────────────────────────
GLUE_TASKS: dict[str, TaskConfig] = {
    "sst2": TaskConfig(
        name="sst2",
        text_fields=("sentence",),
        num_labels=2,
        metric="accuracy",
        eval_metrics=("accuracy",),
        epochs=3,
        description="Sentiment Analysis (SST-2)",
    ),
    "mrpc": TaskConfig(
        name="mrpc",
        text_fields=("sentence1", "sentence2"),
        num_labels=2,
        metric="f1",
        eval_metrics=("f1", "accuracy"),
        epochs=5,
        description="Paraphrase Detection (MRPC)",
    ),
    "qqp": TaskConfig(
        name="qqp",
        text_fields=("question1", "question2"),
        num_labels=2,
        metric="f1",
        eval_metrics=("f1", "accuracy"),
        epochs=3,
        description="Question Pair Equivalence (QQP)",
    ),
    "qnli": TaskConfig(
        name="qnli",
        text_fields=("question", "sentence"),
        num_labels=2,
        metric="accuracy",
        eval_metrics=("accuracy",),
        epochs=3,
        description="Question Natural Language Inference (QNLI)",
    ),
    "rte": TaskConfig(
        name="rte",
        text_fields=("sentence1", "sentence2"),
        num_labels=2,
        metric="accuracy",
        eval_metrics=("accuracy",),
        epochs=5,
        description="Recognizing Textual Entailment (RTE)",
    ),
    "mnli": TaskConfig(
        name="mnli",
        text_fields=("premise", "hypothesis"),
        num_labels=3,
        metric="accuracy",
        eval_metrics=("accuracy",),
        eval_splits=("validation_matched", "validation_mismatched"),
        epochs=3,
        description="Multi-Genre Natural Language Inference (MNLI)",
    ),
    "stsb": TaskConfig(
        name="stsb",
        text_fields=("sentence1", "sentence2"),
        num_labels=1,
        problem_type="regression",
        metric="pearson",
        eval_metrics=("pearson", "spearmanr"),
        epochs=5,
        description="Semantic Textual Similarity (STS-B)",
    ),
}


def get_task(name: str) -> TaskConfig:
    """Look up a task by name, with a clear error if missing."""
    key = name.lower()
    if key not in GLUE_TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(GLUE_TASKS)}")
    return GLUE_TASKS[key]
