"""Squared-composition softmax + degree-reduction analysis (Phase 2).

The current LPAN softmax is a single high-degree (typically 8-12)
polynomial fit to ``exp(x)`` on a profiled interval, evaluated under FHE
via Paterson-Stockmeyer at depth ⌈log₂(deg)⌉+1 (degree 8 ⇒ depth 4,
degree 12 ⇒ depth 4).

This module implements an alternative: the **squared composition**

    exp(x) ≈ ( 1 + p_low(x) / 2^k )^(2^k)

where ``p_low`` is a low-degree polynomial (typically 1–3) approximating
``log(1 + exp(x)/2^k)`` over the same interval, and the outer ``2^k`` is
realised by ``k`` consecutive ciphertext squarings.  The total
multiplicative depth is ``ceil(log2(deg(p_low))) + k``; for ``deg(p_low)
= 2`` and ``k = 2`` we get depth 3 (same as direct degree-4 polyval) but
match a degree-8 fit in expressive power because of the outer expansion.

Public API
----------
* :func:`squared_composition_fit` — fit ``p_low`` for given ``k``.
* :func:`squared_composition_eval` — plaintext evaluator for sanity.
* :func:`compare_softmax_variants` — produce a (degree, depth, error)
  decision table comparing direct-degree fits and squared compositions
  against a reference high-degree Chebyshev fit.

This module **does not** wire anything into the encryption protocol;
its purpose is to give us the data needed to decide whether retraining
LPAN with a different softmax is worth the engineering cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .approximation import (
    chebyshev_approx,
    eval_chebyshev,
    exp_func,
    weighted_minimax_approx,
)


Interval = Tuple[float, float]


# ──────────────────────────────────────────────────────────────────────
# Squared-composition softmax
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SquaredCompositionFit:
    """Result of fitting ``exp(x) ≈ (1 + p_low(x)/2^k)^(2^k)``."""

    inner_power_coeffs: np.ndarray   # power-basis coeffs of p_low
    inner_degree: int
    k_squarings: int
    interval: Interval

    @property
    def effective_degree(self) -> int:
        """Algebraic degree of the expanded polynomial."""
        return self.inner_degree * (1 << self.k_squarings)

    @property
    def multiplicative_depth(self) -> int:
        """CKKS depth: PS for ``p_low`` + k squarings + 1 (the constant add)."""
        ps_depth = max(1, math.ceil(math.log2(max(self.inner_degree, 2))))
        # +1 for the (1 + p/2^k) shift+scale baked into the leaf, +k squarings.
        return ps_depth + self.k_squarings


def squared_composition_fit(
    interval: Interval,
    inner_degree: int,
    k_squarings: int,
    num_points: int = 2000,
    weights_density=None,
) -> SquaredCompositionFit:
    """Fit ``p_low`` so that ``(1 + p_low(x)/2^k)^(2^k) ≈ exp(x)``.

    Strategy: take the ``2^k``-th root of ``exp(x) = exp(x/2^k)^(2^k)``,
    so we need ``1 + p_low(x)/2^k ≈ exp(x/2^k)``, i.e.
    ``p_low(x) ≈ 2^k · (exp(x/2^k) − 1)``.  This is much smoother than
    ``exp(x)`` itself, so a low-degree polynomial suffices.
    """
    a, b = interval
    N = 1 << k_squarings
    target = lambda x: float(N) * (np.exp(np.asarray(x) / float(N)) - 1.0)

    if weights_density is not None:
        cheb_coeffs, _ = weighted_minimax_approx(
            target, interval, inner_degree, density_func=weights_density,
            num_points=num_points,
        )
    else:
        cheb_coeffs, _ = chebyshev_approx(target, interval, inner_degree)

    # Convert to power basis on the original interval [a, b].
    # eval_chebyshev already maps; we sample-and-fit to get power-basis coeffs.
    xs = np.linspace(a, b, max(8 * (inner_degree + 1), 200))
    ys_inner = eval_chebyshev(cheb_coeffs, interval, xs)
    power_coeffs = np.polyfit(xs, ys_inner, inner_degree)[::-1]  # ascending order

    return SquaredCompositionFit(
        inner_power_coeffs=power_coeffs,
        inner_degree=inner_degree,
        k_squarings=k_squarings,
        interval=interval,
    )


def squared_composition_eval(fit: SquaredCompositionFit, x: np.ndarray) -> np.ndarray:
    """Plaintext evaluator: ``(1 + p_low(x)/2^k)^(2^k)``."""
    x = np.asarray(x, dtype=np.float64)
    p = np.zeros_like(x)
    for i, c in enumerate(fit.inner_power_coeffs):
        p = p + c * (x ** i)
    N = 1 << fit.k_squarings
    base = 1.0 + p / float(N)
    out = base
    for _ in range(fit.k_squarings):
        out = out * out
    return out


# ──────────────────────────────────────────────────────────────────────
# Decision-table generator
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantReport:
    name: str
    degree: int
    multiplicative_depth: int
    linf_error: float
    l2_error: float
    relative_l2: float
    notes: str = ""


def _direct_chebyshev_eval(degree: int, interval: Interval, xs: np.ndarray) -> np.ndarray:
    coeffs, _ = chebyshev_approx(exp_func, interval, degree)
    return eval_chebyshev(coeffs, interval, xs)


def compare_softmax_variants(
    interval: Interval = (-6.0, 2.0),
    reference_degree: int = 12,
    direct_degrees: Sequence[int] = (4, 6, 8),
    squared_specs: Sequence[Tuple[int, int]] = ((2, 2), (3, 2), (2, 3)),
    n_eval: int = 4001,
) -> List[VariantReport]:
    """Produce a decision table comparing softmax-poly variants.

    Parameters
    ----------
    interval :
        (a, b) input range. Default mirrors the typical LPAN softmax
        post-shift interval (logits − max).
    reference_degree :
        High-degree direct Chebyshev fit used as the "reference" target.
        Errors are reported against the *true* ``exp(x)`` (not against
        this reference) to keep them physically meaningful.
    direct_degrees :
        Direct-fit degrees to compare.
    squared_specs :
        List of ``(inner_degree, k_squarings)`` pairs to test.

    Returns
    -------
    List[VariantReport], sorted by (multiplicative_depth ascending, linf
    error ascending). The first row whose error is acceptable for the
    application is the recommended choice.
    """
    a, b = interval
    xs = np.linspace(a, b, n_eval)
    y_true = exp_func(xs)
    ref_norm = float(np.sqrt(np.mean(y_true ** 2)) + 1e-12)

    reports: List[VariantReport] = []

    # Reference: high-degree direct Chebyshev (current LPAN style).
    y_ref = _direct_chebyshev_eval(reference_degree, interval, xs)
    err_inf = float(np.max(np.abs(y_true - y_ref)))
    err_l2 = float(np.sqrt(np.mean((y_true - y_ref) ** 2)))
    reports.append(VariantReport(
        name=f"reference_cheb_deg{reference_degree}",
        degree=reference_degree,
        multiplicative_depth=max(1, math.ceil(math.log2(reference_degree))) + 1,
        linf_error=err_inf, l2_error=err_l2, relative_l2=err_l2 / ref_norm,
        notes="current LPAN baseline (PS + affine absorbed)",
    ))

    # Direct lower-degree variants.
    for d in direct_degrees:
        y_d = _direct_chebyshev_eval(d, interval, xs)
        err_inf = float(np.max(np.abs(y_true - y_d)))
        err_l2 = float(np.sqrt(np.mean((y_true - y_d) ** 2)))
        reports.append(VariantReport(
            name=f"direct_cheb_deg{d}",
            degree=d,
            multiplicative_depth=max(1, math.ceil(math.log2(max(d, 2)))) + 1,
            linf_error=err_inf, l2_error=err_l2, relative_l2=err_l2 / ref_norm,
            notes=f"degree-{d} direct fit; lower depth, higher error",
        ))

    # Squared-composition variants.
    for inner_deg, k in squared_specs:
        try:
            fit = squared_composition_fit(interval, inner_deg, k)
        except Exception as exc:
            reports.append(VariantReport(
                name=f"squared_p{inner_deg}_k{k}",
                degree=inner_deg * (1 << k),
                multiplicative_depth=-1,
                linf_error=float("inf"), l2_error=float("inf"), relative_l2=float("inf"),
                notes=f"FIT FAILED: {exc}",
            ))
            continue
        y_sq = squared_composition_eval(fit, xs)
        err_inf = float(np.max(np.abs(y_true - y_sq)))
        err_l2 = float(np.sqrt(np.mean((y_true - y_sq) ** 2)))
        reports.append(VariantReport(
            name=f"squared_p{inner_deg}_k{k}",
            degree=fit.effective_degree,
            multiplicative_depth=fit.multiplicative_depth,
            linf_error=err_inf, l2_error=err_l2, relative_l2=err_l2 / ref_norm,
            notes=f"(1+p/{1<<k})^{1<<k}; effective degree {fit.effective_degree}",
        ))

    reports.sort(key=lambda r: (r.multiplicative_depth, r.linf_error))
    return reports


def format_report(reports: Sequence[VariantReport]) -> str:
    """Pretty-print a comparison table."""
    lines = [
        f"{'variant':<28} {'deg':>4} {'depth':>5} {'L_inf':>10} {'L2':>10} {'rel_L2':>9}  notes",
        "-" * 110,
    ]
    for r in reports:
        lines.append(
            f"{r.name:<28} {r.degree:>4} {r.multiplicative_depth:>5} "
            f"{r.linf_error:>10.4g} {r.l2_error:>10.4g} {r.relative_l2:>9.2%}  {r.notes}"
        )
    return "\n".join(lines)
