"""FHE latency profiling for HyPER-LPAN thesis benchmarks.

Companion to ``accuracy.py``.  Wraps the existing
``encrypt_inference``/``encrypt_inference_hybrid`` paths with a
per-operation timing breakdown suitable for paper tables.

Design notes
------------
The slow path (OpenFHE keygen + bootstrapping + ct×ct multiplications)
is identical to ``experiments/run_fhe_benchmark.py``.  This module is
the importable, scriptable equivalent: tests + ablations should call
``profile_latency`` directly instead of shelling out.

Public API
----------
profile_latency(...)
    Run N samples through the chosen encrypted protocol; return
    a per-operation timing summary plus per-sample wall times.

aggregate_timings(samples) -> dict
    Reduce a list of per-sample timing dicts to mean / median / std
    per operation.  Useful for table generation in notebooks.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LatencyResult:
    """One profiling run summary."""
    variant: str                # "lpan" | "hybrid" | "linear_mixing"
    model_key: str
    task: str
    n_samples: int
    max_seq_len: int
    n_jobs: int
    keygen_time_s: float
    per_sample_wall_s: List[float]
    mean_op_timings_s: Dict[str, float]
    median_op_timings_s: Dict[str, float]
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_latency_s(self) -> float:
        return float(np.mean(self.per_sample_wall_s)) if self.per_sample_wall_s else 0.0

    @property
    def median_latency_s(self) -> float:
        return float(np.median(self.per_sample_wall_s)) if self.per_sample_wall_s else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_latency_s"] = self.mean_latency_s
        d["median_latency_s"] = self.median_latency_s
        return d


def aggregate_timings(samples: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample timing dicts to {op: {mean, median, std}}."""
    if not samples:
        return {}
    keys = set()
    for s in samples:
        keys.update(s.keys())
    keys.discard("total")
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = np.array([float(s.get(k, 0.0)) for s in samples], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
        }
    return out


def _make_backend(mult_depth: int, ring_dim: int, enable_bootstrap: bool, n_jobs: int):
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend
    return OpenFHEBackend(
        multiplicative_depth=mult_depth,
        ring_dim=ring_dim,
        enable_bootstrap=enable_bootstrap,
        num_threads=n_jobs if n_jobs > 0 else (os.cpu_count() or 1),
    )


def profile_latency(
    model_key: str,
    task: str,
    *,
    variant: str = "hybrid",
    checkpoint_path: str | Path,
    embeddings: Sequence[np.ndarray],
    max_seq_len: int = 64,
    n_jobs: int = 1,
    mult_depth: int = 25,
    ring_dim: int = 1 << 16,
    enable_bootstrap: bool = True,
    linear_mixing_layers: Sequence[int] = (0, 1, 2, 3),
    quad_attention_layers: Sequence[int] = (4, 5, 6, 7),
    reduced_degrees: bool = False,
    kept_token_indices_per_sample: Sequence[np.ndarray] | None = None,
    word_elimination: str = "none",
) -> LatencyResult:
    """Run encrypted inference on ``embeddings`` and return latency stats.

    ``embeddings`` is a sequence of (seq_len, hidden) numpy arrays — the
    plaintext embedding-layer output for each sample (already padded to
    ``max_seq_len``).  Compute with ``embeddings.py``-style helpers in
    ``experiments/run_fhe_benchmark.py``.
    """
    from fhe_thesis.encryption.coefficients import load_coefficients
    from fhe_thesis.encryption.protocol import (
        encrypt_inference,
        encrypt_inference_hybrid,
        encrypt_inference_linear_mixing,
        load_hybrid_weights,
        load_linear_mixing_weights,
        load_model_weights,
    )

    t0 = time.time()
    backend = _make_backend(mult_depth, ring_dim, enable_bootstrap, n_jobs)
    keygen = time.time() - t0

    if variant == "hybrid":
        weights = load_hybrid_weights(
            model_key,
            checkpoint_path=str(checkpoint_path),
            linear_mixing_layers=list(linear_mixing_layers),
            quad_attention_layers=list(quad_attention_layers),
        )
        if reduced_degrees:
            from fhe_thesis.encryption.hybrid_coefficients import load_coefficients_for_hybrid
            coeffs = load_coefficients_for_hybrid(
                model_key, task=task,
                linear_mixing_layers=list(linear_mixing_layers),
                quad_attention_layers=list(quad_attention_layers),
            )
        else:
            coeffs = load_coefficients(model_key, task=task)
            for li in list(linear_mixing_layers) + list(quad_attention_layers):
                if li in coeffs:
                    coeffs[li] = {k: v for k, v in coeffs[li].items() if k != "Softmax"}
        run = lambda emb, ki=None: encrypt_inference_hybrid(
            backend, emb, weights, coeffs,
            max_seq_len=max_seq_len, n_jobs=n_jobs,
            kept_token_indices=ki,
        )
    elif variant == "linear_mixing":
        weights = load_linear_mixing_weights(model_key, checkpoint_path=str(checkpoint_path))
        coeffs = load_coefficients(model_key, task=task)
        run = lambda emb, ki=None: encrypt_inference_linear_mixing(
            backend, emb, weights, coeffs,
            max_seq_len=max_seq_len, n_jobs=n_jobs,
            kept_token_indices=ki,
        )
    elif variant == "lpan":
        weights = load_model_weights(model_key, checkpoint_path=str(checkpoint_path))
        coeffs = load_coefficients(model_key, task=task)
        run = lambda emb, ki=None: encrypt_inference(
            backend, emb, weights, coeffs,
            max_seq_len=max_seq_len, n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    per_sample_wall: List[float] = []
    per_sample_timings: List[Dict[str, float]] = []
    for i, emb in enumerate(embeddings):
        ki = (kept_token_indices_per_sample[i]
              if kept_token_indices_per_sample is not None else None)
        t = time.time()
        _logits, timings = run(emb, ki) if variant in ("hybrid", "linear_mixing") else run(emb)
        per_sample_wall.append(time.time() - t)
        per_sample_timings.append(dict(timings))

    agg = aggregate_timings(per_sample_timings)
    mean_ops = {k: v["mean"] for k, v in agg.items()}
    median_ops = {k: v["median"] for k, v in agg.items()}

    return LatencyResult(
        variant=variant,
        model_key=model_key,
        task=task,
        n_samples=len(embeddings),
        max_seq_len=max_seq_len,
        n_jobs=n_jobs,
        keygen_time_s=keygen,
        per_sample_wall_s=per_sample_wall,
        mean_op_timings_s=mean_ops,
        median_op_timings_s=median_ops,
        config={
            "mult_depth": mult_depth,
            "ring_dim": ring_dim,
            "enable_bootstrap": enable_bootstrap,
            "reduced_degrees": reduced_degrees,
            "linear_mixing_layers": list(linear_mixing_layers) if variant == "hybrid" else None,
            "quad_attention_layers": list(quad_attention_layers) if variant == "hybrid" else None,
            "checkpoint_path": str(checkpoint_path),
            "word_elimination": word_elimination,
        },
    )
