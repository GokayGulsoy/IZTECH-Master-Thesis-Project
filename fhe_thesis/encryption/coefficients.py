"""LPAN polynomial coefficient loading.

Two sources, in priority order:

1. **Trained LPAN checkpoint** — coefficients learned by
   ``run_staged_lpan.py`` and dumped to ``results/coefficients/<model>.json``
   by ``extract_coefficients.py``. Always preferred when available.

2. **Profile-and-fit fallback** — runs ``profile_model`` on a few
   hundred samples and fits a weighted-minimax polynomial. Useful for
   smoke-tests on a model you have not yet LPAN-trained.

The returned object is a ``dict[layer_idx, dict[op, PolyCoeffs]]``
where ``op ∈ {"GELU", "Softmax", "LN"}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from fhe_thesis.config import (
    MODEL_REGISTRY,
    PROFILED_INTERVALS,
    RESULTS_DIR,
)
from fhe_thesis.poly.approximation import (
    exp_func,
    gelu_func,
    inv_sqrt_func,
    weighted_minimax_approx,
)
from fhe_thesis.poly.chebyshev import chebyshev_to_power


@dataclass(frozen=True)
class PolyCoeffs:
    """One polynomial fit: power-basis coefficients + approximation interval.

    For per-head softmax, ``power_coeffs`` is a 2D array of shape
    ``(num_heads, degree+1)`` and ``per_head`` is True.
    """

    power_coeffs: np.ndarray  # 1D for GELU/LN, 2D for per-head softmax
    interval: tuple[float, float]
    degree: int
    per_head: bool = False


# ── Public API ────────────────────────────────────────────────────────


def load_coefficients(
    model_key: str,
    *,
    task: str = "sst2",
    degree: int = 8,
    profile_samples: int = 400,
    coefficients_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, PolyCoeffs]]:
    """Return ``{layer_idx: {op_name: PolyCoeffs}}`` for ``(model_key, task)``.

    Tries the LPAN checkpoint dump first, falls back to profile-and-fit.
    All tasks use the unified ``results/coefficients/<task>/bert_<model>_coeffs.json``
    layout. A legacy flat-path fallback exists for SST-2 files created before
    the layout change.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {model_key!r}; " f"options: {sorted(MODEL_REGISTRY)}"
        )

    base_dir = coefficients_dir or RESULTS_DIR / "coefficients"
    coeff_path = base_dir / task / f"bert_{model_key}_coeffs.json"
    # Legacy flat layout fallback for SST-2 (pre-unified layout)
    if not coeff_path.exists() and task == "sst2":
        coeff_path = base_dir / f"bert_{model_key}_coeffs.json"
    if coeff_path.exists():
        return _load_from_extracted(coeff_path)

    return _profile_and_fit(model_key, degree=degree, num_samples=profile_samples)


# ── Source 1: extracted LPAN checkpoint ───────────────────────────────


def _load_from_extracted(path: Path) -> Dict[int, Dict[str, PolyCoeffs]]:
    """Parse the flat JSON written by ``extract_coefficients.py``.

    Shape: ``{param_name: {activation_type, layer, degree, coefficients}}``.
    The checkpoint does not store the approximation interval; we look it
    up in :data:`PROFILED_INTERVALS` and fall back to the L0 entry when
    a layer-specific interval is missing (e.g. layer ≥ 2 for BERT-Base).
    """
    raw = json.loads(path.read_text())
    op_map = {"gelu": "GELU", "softmax": "Softmax", "layernorm": "LN"}
    out: Dict[int, Dict[str, PolyCoeffs]] = {}
    for entry in raw.values():
        act = entry.get("activation_type")
        layer_idx = entry.get("layer")
        if act not in op_map or layer_idx is None:
            continue
        op_name = op_map[act]
        raw_coeffs = entry["coefficients"]
        is_per_head = entry.get("num_heads") is not None
        cheb = np.asarray(raw_coeffs, dtype=np.float64)
        # Trained checkpoints store *Chebyshev* coefficients; the
        # encrypted side evaluates polynomials in the *power* basis
        # (TenSEAL ``polyval``), so convert here once on load.
        if is_per_head:
            coeffs = np.stack(
                [np.asarray(chebyshev_to_power(row), dtype=np.float64)
                 for row in cheb]
            )
        else:
            coeffs = np.asarray(chebyshev_to_power(cheb), dtype=np.float64)
        interval = PROFILED_INTERVALS.get(
            f"L{layer_idx}_{op_name}",
            PROFILED_INTERVALS[f"L0_{op_name}"],
        )
        degree = entry.get("degree", coeffs.shape[-1] - 1)
        out.setdefault(layer_idx, {})[op_name] = PolyCoeffs(
            power_coeffs=coeffs,
            interval=(float(interval[0]), float(interval[1])),
            degree=degree,
            per_head=is_per_head,
        )
    return out


# ── Source 2: profile-and-fit fallback ────────────────────────────────


def _profile_and_fit(
    model_key: str,
    *,
    degree: int,
    num_samples: int,
) -> Dict[int, Dict[str, PolyCoeffs]]:
    from fhe_thesis.models.profiling import build_kde_density, profile_model

    cfg = MODEL_REGISTRY[model_key]
    profile = profile_model(
        cfg["name"], num_layers=cfg["layers"], num_samples=num_samples
    )

    func_map = {"GELU": gelu_func, "Softmax": exp_func, "LN": inv_sqrt_func}
    density_keys = {
        "GELU": "gelu_inputs",
        "Softmax": "softmax_inputs",
        "LN": "ln_variances",
    }

    out: Dict[int, Dict[str, PolyCoeffs]] = {}
    for layer_idx in range(cfg["layers"]):
        out[layer_idx] = {}
        for op_name, func in func_map.items():
            interval = PROFILED_INTERVALS.get(
                f"L{layer_idx}_{op_name}",
                # Fall back to L0_* intervals for layers we have no
                # profile for (e.g. BERT-Base layer 7).
                PROFILED_INTERVALS[f"L0_{op_name}"],
            )
            dk = density_keys[op_name]
            if dk in profile and layer_idx in profile[dk]:
                density = build_kde_density(profile[dk][layer_idx])
            else:
                density = lambda x: np.ones_like(x, dtype=float)
            cheb_c, _ = weighted_minimax_approx(func, interval, degree, density)
            power_c = chebyshev_to_power(cheb_c)
            out[layer_idx][op_name] = PolyCoeffs(
                power_coeffs=np.asarray(power_c, dtype=np.float64),
                interval=interval,
                degree=degree,
            )
    return out
