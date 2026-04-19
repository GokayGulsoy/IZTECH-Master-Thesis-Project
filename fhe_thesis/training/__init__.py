"""Training utilities: metrics, data loading, fine-tuning, and distillation."""
from .trainer import (
    NaNSafeTrainer,
    DistillationTrainer,
    calibrate_grad_norm,
    compute_metrics,
    detect_device,
    distill_and_eval,
    load_sst2_dataset,
    train_and_eval,
)
