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
    # Softmax operates on shifted scores ≤ 0; exp(-15) ≈ 3e-7 is already
    # negligible relative to slots near 0. Capping at -15 prevents
    # degree-12 Chebyshev approximations of `exp` on absurdly wide
    # intervals (where the polynomial oscillates wildly between sample
    # points and produces meaningless decryptions).
    "Softmax": (-15.0, 0.5),
    "LN": (0.01, 50.0),
}

# ── Fallback approximation intervals ─────────────────────────────────────────
FALLBACK_INTERVALS: Dict[str, Interval] = {
    "GELU": (-5.0, 5.0),
    "Softmax": (-8.0, 0.0),
    "LN": (0.1, 4.0),
}


# ── LPAN-Hybrid checkpoint schedules (PBRP) ─────────────────────────────────
# Per-model lists of (layer_idx, position) where position ∈ {'mid', 'end'}.
# 'mid'  → checkpoint between attention and FFN of that layer
# 'end'  → checkpoint after FFN of that layer
# Empty list ⇒ pure-FHE (k=0). Heavier schedules trade interactivity for
# wall-time / depth budget. See thesis chapter 3.
CHECKPOINT_SCHEDULES: Dict[str, Dict[int, list]] = {
    "tiny": {
        0: [],                              # k=0: pure FHE baseline
        1: [(0, "end")],                    # k=1: 1 chkpt
        2: [(0, "mid"), (0, "end"),
            (1, "mid"), (1, "end")],        # k=2: 4 chkpts
    },
    "mini": {
        0: [],
        1: [(i, "end") for i in range(4)],
        2: [(i, p) for i in range(4) for p in ("mid", "end")],
    },
    "small": {
        0: [],
        1: [(i, "end") for i in range(4)],
        2: [(i, p) for i in range(4) for p in ("mid", "end")],
    },
    "base": {
        0: [],
        1: [(i, "end") for i in (0, 3, 6, 9)],   # every 3 layers
        2: [(i, "end") for i in range(0, 12, 2)],  # every 2 layers
    },
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
