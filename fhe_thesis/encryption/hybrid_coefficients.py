"""Hybrid-aware polynomial coefficient selection for HyPER-LPAN.

Standard ``load_coefficients`` returns coefficients for every (layer, op)
triple in the trained checkpoint.  In a hybrid composition many of those
entries are wasted — LinearMixing and 2Quad layers do not invoke the
softmax polynomial at all, and the surviving LPAN layers can often run
with a lower-degree polynomial than the LPAN training run.

This module adds two helpers:

* :func:`required_ops_for_layer` — given a layer index and the hybrid
  composition, returns the set of polynomial ops actually invoked at
  inference time (a subset of ``{"GELU", "Softmax", "LN"}``).
* :func:`load_coefficients_for_hybrid` — wraps :func:`load_coefficients`
  and drops every entry not in the required set, optionally re-fitting a
  lower-degree polynomial on the profile data.

Re-fitting note
---------------
When ``softmax_degree`` is provided and is strictly less than the
trained checkpoint's degree, we **truncate** the existing coefficients
(power-basis truncation).  This is suboptimal vs running
``weighted_minimax_approx`` from scratch on the profile data, but is
zero-cost and adequate for a first-pass depth budget audit.  To re-fit
properly use the standalone script
``scripts/refit_hybrid_polynomials.py`` (which calls into
``fhe_thesis.poly.approximation``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Dict, Optional, Sequence, Set

import numpy as np

from fhe_thesis.encryption.coefficients import PolyCoeffs, load_coefficients


# Polynomial ops by attention region.
_OPS_BY_REGION: Dict[str, Set[str]] = {
    "linear_mixing": {"GELU", "LN"},      # No softmax; LN is post-attn + post-FFN
    "quad":          {"GELU", "LN"},      # No softmax (replaced by squaring)
    "lpan":          {"GELU", "Softmax", "LN"},
}


def classify_layer(
    layer_idx: int,
    linear_mixing_layers: Sequence[int],
    quad_attention_layers: Sequence[int],
) -> str:
    """Return ``"linear_mixing"`` | ``"quad"`` | ``"lpan"`` for ``layer_idx``."""
    if layer_idx in linear_mixing_layers:
        return "linear_mixing"
    if layer_idx in quad_attention_layers:
        return "quad"
    return "lpan"


def required_ops_for_layer(
    layer_idx: int,
    linear_mixing_layers: Sequence[int],
    quad_attention_layers: Sequence[int],
) -> Set[str]:
    """Polynomial ops actually invoked by ``encrypt_layer_dispatch`` at this layer."""
    region = classify_layer(layer_idx, linear_mixing_layers, quad_attention_layers)
    return _OPS_BY_REGION[region]


def truncate_to_degree(coeffs: PolyCoeffs, new_degree: int) -> PolyCoeffs:
    """Return a copy of ``coeffs`` truncated to ``new_degree`` (power basis).

    For ``new_degree >= coeffs.degree`` this is a no-op (no zero-padding —
    the caller is asking for at least the existing precision).
    """
    if new_degree >= coeffs.degree:
        return coeffs
    if coeffs.per_head:
        # Shape (num_heads, degree+1) → keep first new_degree+1 columns
        new_pc = coeffs.power_coeffs[:, : new_degree + 1].copy()
    else:
        new_pc = coeffs.power_coeffs[: new_degree + 1].copy()
    return replace(coeffs, power_coeffs=new_pc, degree=new_degree)


def load_coefficients_for_hybrid(
    model_key: str,
    *,
    task: str = "sst2",
    linear_mixing_layers: Sequence[int] = (),
    quad_attention_layers: Sequence[int] = (),
    softmax_degree: Optional[int] = None,
    gelu_degree: Optional[int] = None,
    ln_degree: Optional[int] = None,
    base_degree: int = 8,
    profile_samples: int = 400,
) -> Dict[int, Dict[str, PolyCoeffs]]:
    """Load LPAN coefficients pruned + truncated for a hybrid composition.

    Parameters
    ----------
    model_key, task : str
        Forwarded to :func:`load_coefficients`.
    linear_mixing_layers, quad_attention_layers : sequences of int
        The hybrid composition.  Layers not in either are LPAN.
    softmax_degree, gelu_degree, ln_degree : int, optional
        Per-op degree caps.  ``None`` keeps the trained-checkpoint degree.
        Truncation is applied only when ``new_degree < trained_degree``.
    base_degree, profile_samples
        Forwarded to :func:`load_coefficients` for the profile-and-fit
        fallback path.

    Returns
    -------
    dict
        ``{layer_idx: {op_name: PolyCoeffs}}`` — entries for unused ops
        are omitted entirely (e.g. linear-mixing layers have no
        ``"Softmax"`` key).
    """
    base = load_coefficients(
        model_key,
        task=task,
        degree=base_degree,
        profile_samples=profile_samples,
    )

    deg_overrides = {
        "GELU":    gelu_degree,
        "Softmax": softmax_degree,
        "LN":      ln_degree,
    }

    out: Dict[int, Dict[str, PolyCoeffs]] = {}
    for layer_idx, ops in base.items():
        needed = required_ops_for_layer(
            layer_idx, linear_mixing_layers, quad_attention_layers,
        )
        out[layer_idx] = {}
        for op_name, coeffs in ops.items():
            if op_name not in needed:
                continue
            new_deg = deg_overrides.get(op_name)
            if new_deg is not None:
                coeffs = truncate_to_degree(coeffs, new_deg)
            out[layer_idx][op_name] = coeffs
    return out


def summarize_hybrid_coeffs(
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
    linear_mixing_layers: Sequence[int],
    quad_attention_layers: Sequence[int],
) -> str:
    """Human-readable summary of which ops/degrees survive per layer."""
    lines = ["Layer | region        | ops (degree)"]
    lines.append("-" * 52)
    for li in sorted(coeffs):
        region = classify_layer(li, linear_mixing_layers, quad_attention_layers)
        ops_str = ", ".join(
            f"{op}={c.degree}" for op, c in sorted(coeffs[li].items())
        )
        lines.append(f"  L{li:<3d} | {region:<13s} | {ops_str}")
    return "\n".join(lines)
