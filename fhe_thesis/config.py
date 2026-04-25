"""Shared configuration: model registry, constants, paths, intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

# ── Type aliases ──────────────────────────────────────────────────────────────
Interval = Tuple[float, float]
ChebResult = Tuple[np.ndarray, Interval]
DensityFunc = Callable[[np.ndarray], np.ndarray]
ArrayLike = Union[np.ndarray, float]

# ── Output directories ───────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
POLY_APPROX_DIR = RESULTS_DIR / "poly_approx"
ACTIVATION_PROFILES_DIR = RESULTS_DIR / "activation_profiles"
DEPTH_ALLOCATION_DIR = RESULTS_DIR / "depth_allocation"
BSGS_EVAL_DIR = RESULTS_DIR / "bsgs_eval"
ENCRYPTED_INFERENCE_DIR = RESULTS_DIR / "encrypted_inference"
ERROR_PROPAGATION_DIR = RESULTS_DIR / "error_propagation"
GA_OPTIMIZATION_DIR = RESULTS_DIR / "ga_optimization"
TRAINING_DIR = RESULTS_DIR / "training"
MODELS_DIR = RESULTS_DIR / "models"
MULTI_MODEL_DIR = RESULTS_DIR / "multi_model"
MULTI_DATASET_DIR = RESULTS_DIR / "multi_dataset"

# ── Model registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "name": "google/bert_uncased_L-2_H-128_A-2",
        "short": "BERT-Tiny",
        "layers": 2, "hidden": 128, "heads": 2, "params_m": 4.4,
        "batch_size": 32, "lr": 3e-5,
    },
    "mini": {
        "name": "google/bert_uncased_L-4_H-256_A-4",
        "short": "BERT-Mini",
        "layers": 4, "hidden": 256, "heads": 4, "params_m": 11.2,
        "batch_size": 32, "lr": 3e-5,
    },
    "small": {
        "name": "google/bert_uncased_L-4_H-512_A-8",
        "short": "BERT-Small",
        "layers": 4, "hidden": 512, "heads": 8, "params_m": 28.8,
        "batch_size": 16, "lr": 2e-5,
    },
    "base": {
        "name": "bert-base-uncased",
        "short": "BERT-Base",
        "layers": 12, "hidden": 768, "heads": 12, "params_m": 110.0,
        "batch_size": 16, "lr": 2e-5,
    },
}

# Default model for single-model experiments
DEFAULT_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"

# ── Safe maximum intervals (prevents polynomial blow-up on outliers) ─────────
MAX_INTERVALS: Dict[str, Interval] = {
    "GELU": (-10.0, 10.0),
    "Softmax": (-20.0, 0.5),  # shifted scores: always ≤ 0 (+ small margin)
    "LN": (0.01, 10.0),    # capped at 10: typical BERT variance is [0.1, 8]; degree-4 poly can fit 1/√x accurately over this range
}

# ── Fallback approximation intervals ─────────────────────────────────────────
FALLBACK_INTERVALS: Dict[str, Interval] = {
    "GELU": (-5.0, 5.0),
    "Softmax": (-8.0, 0.0),
    "LN": (0.1, 4.0),
}

# ── Profiled intervals from BERT-Tiny (used by GA, error propagation) ────────
PROFILED_INTERVALS: Dict[str, Interval] = {
    "L0_GELU": (-6.365, 4.173),
    "L0_Softmax": (-2.101, 4.523),
    "L0_LN": (0.905, 6.714),
    "L1_GELU": (-4.712, 2.494),
    "L1_Softmax": (-7.175, 8.401),
    "L1_LN": (0.902, 7.295),
}

# ── Function/density key mapping ─────────────────────────────────────────────
FUNC_NAMES = ["GELU", "Softmax", "LN"]
PROFILE_KEY_MAP: Dict[str, str] = {
    "GELU": "gelu_inputs",
    "Softmax": "softmax_inputs",
    "LN": "ln_variances",
}


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [POLY_APPROX_DIR, ACTIVATION_PROFILES_DIR, DEPTH_ALLOCATION_DIR,
              BSGS_EVAL_DIR, ENCRYPTED_INFERENCE_DIR, ERROR_PROPAGATION_DIR,
              GA_OPTIMIZATION_DIR, TRAINING_DIR, MODELS_DIR, MULTI_MODEL_DIR,
              MULTI_DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)
