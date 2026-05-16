"""Polynomial approximation methods for FHE-compatible activation functions.

Consolidates Taylor, Chebyshev, Least-Squares, and Weighted Minimax
approximation for GELU, exp, and inverse square root.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.polynomial import Polynomial
from scipy.special import erf

from ..config import (
    ArrayLike, ChebResult, DensityFunc, Interval,
    POLY_APPROX_DIR,
)


# ── Target functions ──────────────────────────────────────────────────────────

def gelu_func(x: ArrayLike) -> ArrayLike:
    """Exact GELU: x/2 * [1 + erf(x/sqrt(2))]."""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def exp_func(x: ArrayLike) -> ArrayLike:
    """Exponential function (for Softmax), clipped for numerical safety."""
    return np.exp(np.clip(x, -20, 20))


def inv_sqrt_func(x: ArrayLike) -> ArrayLike:
    """Inverse square root (for LayerNorm), with epsilon for safety."""
    return 1.0 / np.sqrt(np.maximum(x, 1e-10))


# ── Density functions ─────────────────────────────────────────────────────────

def gaussian_density(center: float = 0.0, std: float = 1.5) -> DensityFunc:
    """Gaussian density centred at *center*."""
    def density(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - center) / std) ** 2)
    return density


def shifted_exp_density(shift: float = -4.0, scale: float = 2.0) -> DensityFunc:
    """Shifted Gaussian density for attention-logit distributions."""
    def density(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - shift) / scale) ** 2)
    return density


def variance_density(mean: float = 1.0, std: float = 0.5) -> DensityFunc:
    """Half-Gaussian density for LayerNorm variance values."""
    def density(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) / std) ** 2) * (x > 0).astype(float)
    return density


# ── Approximation methods ─────────────────────────────────────────────────────

def taylor_approx(
    func: Callable, interval: Interval, degree: int, center: float = 0.0,
) -> Polynomial:
    """Taylor polynomial of *func* up to *degree* about *center*."""
    h = 1e-4
    coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        coeffs[k] = _finite_diff_derivative(func, center, k, h=h) / math.factorial(k)
    return Polynomial(coeffs)


def _finite_diff_derivative(func: Callable, x0: float, order: int, h: float = 1e-4) -> float:
    if order == 0:
        return float(func(x0))
    stencil = _fd_coefficients(order)
    n = len(stencil)
    result = 0.0
    for i, c in enumerate(stencil):
        result += c * float(func(x0 + (i - n // 2) * h))
    return result / (h ** order)


def _fd_coefficients(order: int) -> np.ndarray:
    table = {
        1: np.array([-0.5, 0.0, 0.5]),
        2: np.array([1.0, -2.0, 1.0]),
        3: np.array([-0.5, 1.0, 0.0, -1.0, 0.5]),
        4: np.array([1.0, -4.0, 6.0, -4.0, 1.0]),
        5: np.array([-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5]),
        6: np.array([1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]),
        7: np.array([-0.5, 3.0, -7.0, 7.0, 0.0, -7.0, 7.0, -3.0, 0.5]),
        8: np.array([1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0]),
        9: np.array([-0.5, 4.0, -13.5, 24.0, -17.5, 0.0, 17.5, -24.0, 13.5, -4.0, 0.5]),
    }
    if order in table:
        return table[order]
    raise ValueError(f"Finite difference coefficients not precomputed for order {order}")


def chebyshev_approx(func: Callable, interval: Interval, degree: int) -> ChebResult:
    """Near-minimax Chebyshev interpolation at Chebyshev nodes."""
    a, b = interval
    k = np.arange(degree + 1)
    nodes_std = np.cos((2 * k + 1) * np.pi / (2 * (degree + 1)))
    nodes = 0.5 * (b - a) * nodes_std + 0.5 * (a + b)
    values = func(nodes)
    cheb_coeffs = cheb.chebfit(nodes_std, values, degree)
    return cheb_coeffs, (a, b)


def eval_chebyshev(cheb_coeffs: np.ndarray, interval: Interval, x: ArrayLike) -> ArrayLike:
    """Evaluate Chebyshev polynomial on original interval [a, b]."""
    a, b = interval
    x_std = (2.0 * x - (a + b)) / (b - a)
    return cheb.chebval(x_std, cheb_coeffs)


def least_squares_approx(
    func: Callable, interval: Interval, degree: int,
    num_points: int = 1000, weights: Optional[DensityFunc] = None,
) -> np.poly1d:
    """Fit polynomial by (optionally weighted) least-squares regression."""
    a, b = interval
    x_sample = np.linspace(a, b, num_points)
    y_sample = func(x_sample)
    if weights is not None:
        w = weights(x_sample)
        w = w / w.sum() * len(w)
    else:
        w = np.ones_like(x_sample)
    coeffs = np.polyfit(x_sample, y_sample, degree, w=np.sqrt(w))
    return np.poly1d(coeffs)


def weighted_minimax_approx(
    func: Callable,
    interval: Interval,
    degree: int,
    density_func: DensityFunc,
    num_points: int = 2000,
    remez_iterations: int = 5,
) -> ChebResult:
    """Distribution-weighted minimax polynomial (Contribution 1).

    Solves: argmin_p max_x rho(x) * |f(x) - p(x)|
    using density-weighted least-squares + simplified Remez iterations.
    """
    a, b = interval
    x_dense = np.linspace(a, b, num_points)
    rho = density_func(x_dense)
    rho = rho / (rho.max() + 1e-10)
    rho = np.clip(rho, 0.01, 1.0)

    x_std = (2.0 * x_dense - (a + b)) / (b - a)
    y_target = func(x_dense)

    V = cheb.chebvander(x_std, degree)
    W = np.diag(np.sqrt(rho))
    cheb_coeffs, _, _, _ = np.linalg.lstsq(W @ V, W @ y_target, rcond=None)

    sigma = 0.1 * (b - a)
    for _ in range(remez_iterations):
        p_vals = cheb.chebval(x_std, cheb_coeffs)
        w_err = rho * np.abs(y_target - p_vals)
        worst_idx = int(np.argmax(w_err))
        boost = np.exp(-0.5 * ((x_dense - x_dense[worst_idx]) / sigma) ** 2)
        rho_r = rho + 0.3 * boost
        rho_r = rho_r / rho_r.max()
        rho_r = np.clip(rho_r, 0.01, 1.0)
        W = np.diag(np.sqrt(rho_r))
        cheb_coeffs, _, _, _ = np.linalg.lstsq(W @ V, W @ y_target, rcond=None)

    return cheb_coeffs, (a, b)


# ── Error computation ─────────────────────────────────────────────────────────

def compute_errors(
    func: Callable, poly_eval_func: Callable, x_test: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute L-infinity and L2 (RMSE) approximation errors."""
    y_true = func(x_test)
    y_approx = poly_eval_func(x_test)
    abs_error = np.abs(y_true - y_approx)
    return float(np.max(abs_error)), float(np.sqrt(np.mean(abs_error ** 2))), abs_error


def multiplicative_depth(degree: int) -> int:
    """CKKS multiplicative depth for a polynomial of given degree (balanced tree)."""
    if degree <= 1:
        return degree
    return int(np.ceil(np.log2(degree)))


# ── Comparison pipeline ──────────────────────────────────────────────────────

def compare_approximations(
    func: Callable,
    func_name: str,
    interval: Interval,
    degrees: Sequence[int],
    density_func: Optional[DensityFunc] = None,
    taylor_center: float = 0.0,
    num_test: int = 5000,
    output_dir=None,
) -> Dict[int, Dict[str, Any]]:
    """Compare Taylor, Chebyshev, Least-Squares, and Weighted Minimax."""
    if output_dir is None:
        output_dir = POLY_APPROX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    a, b = interval
    x_test = np.linspace(a, b, num_test)
    x_plot = np.linspace(a, b, 1000)

    if density_func is None:
        density_func = gaussian_density(center=(a + b) / 2, std=(b - a) / 4)

    results: Dict[int, Dict[str, Any]] = {}

    for deg in degrees:
        depth = multiplicative_depth(deg)
        row: Dict[str, Any] = {"degree": deg, "depth": depth}

        # B1: Taylor
        try:
            tp = taylor_approx(func, interval, deg, center=taylor_center)
            linf, l2, _ = compute_errors(func, lambda x, p=tp: p(x), x_test)
            row["taylor"] = {"linf": linf, "l2": l2}
        except Exception as e:
            row["taylor"] = {"linf": float("inf"), "l2": float("inf"), "error": str(e)}

        # B2: Chebyshev
        cc, ci = chebyshev_approx(func, interval, deg)
        linf, l2, _ = compute_errors(func, lambda x, c=cc, i=ci: eval_chebyshev(c, i, x), x_test)
        row["chebyshev"] = {"linf": linf, "l2": l2}

        # B3: Least-squares
        lsp = least_squares_approx(func, interval, deg)
        linf, l2, _ = compute_errors(func, lambda x, p=lsp: p(x), x_test)
        row["least_squares"] = {"linf": linf, "l2": l2}

        # P: Weighted minimax (ours)
        wc, wi = weighted_minimax_approx(func, interval, deg, density_func)
        linf, l2, _ = compute_errors(func, lambda x, c=wc, i=wi: eval_chebyshev(c, i, x), x_test)
        rho_test = density_func(x_test)
        rho_test = rho_test / rho_test.max()
        wlinf = float(np.max(rho_test * np.abs(func(x_test) - eval_chebyshev(wc, wi, x_test))))
        row["weighted_minimax"] = {"linf": linf, "l2": l2, "weighted_linf": wlinf}

        results[deg] = row

    # ── Generate plots ────────────────────────────────────────────────────
    max_deg = max(degrees)
    y_true = func(x_plot)

    methods = ["taylor", "chebyshev", "least_squares", "weighted_minimax"]
    labels = ["B1: Taylor", "B2: Chebyshev", "B3: Least-Squares", "P: Weighted Minimax (Ours)"]
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]

    # Plot 1: Approximation overlay
    tp = taylor_approx(func, interval, max_deg, center=taylor_center)
    cc, ci = chebyshev_approx(func, interval, max_deg)
    lsp = least_squares_approx(func, interval, max_deg)
    wc, wi = weighted_minimax_approx(func, interval, max_deg, density_func)

    approx_data = [
        ("B1: Taylor", tp(x_plot), "tab:red"),
        ("B2: Chebyshev", eval_chebyshev(cc, ci, x_plot), "tab:blue"),
        ("B3: Least-Squares", lsp(x_plot), "tab:orange"),
        ("P: Weighted Minimax (Ours)", eval_chebyshev(wc, wi, x_plot), "tab:green"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{func_name}: Polynomial Approximations (degree {max_deg})", fontsize=14, fontweight="bold")
    for idx, (label, y_approx, color) in enumerate(approx_data):
        ax = axes[idx // 2, idx % 2]
        ax.plot(x_plot, y_true, "k-", linewidth=1.5, label=f"True {func_name}", alpha=0.7)
        ax.plot(x_plot, y_approx, "--", color=color, linewidth=1.5, label=label)
        ax.fill_between(x_plot, y_true, y_approx, alpha=0.15, color=color)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        y_range = float(np.max(np.abs(y_true))) * 1.5
        ax.set_ylim(-y_range, y_range * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{func_name}_approximations_deg{max_deg}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Error vs degree
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{func_name}: Approximation Error vs. Polynomial Degree", fontsize=14, fontweight="bold")
    markers = ["s", "^", "o", "D"]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        linf_vals = [results[d].get(method, {}).get("linf", np.nan) for d in degrees]
        l2_vals = [results[d].get(method, {}).get("l2", np.nan) for d in degrees]
        lw = 2.5 if method == "weighted_minimax" else 1.5
        ax1.semilogy(degrees, linf_vals, f"-{marker}", color=color, label=label, linewidth=lw, markersize=7)
        ax2.semilogy(degrees, l2_vals, f"-{marker}", color=color, label=label, linewidth=lw, markersize=7)
    ax1.set(xlabel="Polynomial Degree", ylabel="$L^\\infty$ Error", title="$L^\\infty$ Error")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Polynomial Degree", ylabel="$L^2$ Error (RMSE)", title="$L^2$ Error")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{func_name}_error_vs_degree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Pointwise error with density overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    rho_plot = density_func(x_plot)
    rho_plot = rho_plot / rho_plot.max()
    ax.fill_between(x_plot, 0, rho_plot * 0.5, alpha=0.15, color="gray", label="Activation density ρ(x)")
    for y_approx_data, label, color in zip(
        [tp(x_plot), eval_chebyshev(cc, ci, x_plot), lsp(x_plot), eval_chebyshev(wc, wi, x_plot)],
        labels, colors
    ):
        error = np.abs(y_true - y_approx_data)
        ax.semilogy(x_plot, error + 1e-16, "-", color=color, label=label, alpha=0.8, linewidth=1.3)
    ax.set(xlabel="x", ylabel="Absolute Error (log scale)",
           title=f"{func_name}: Pointwise Error at Degree {max_deg} with Activation Density")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{func_name}_pointwise_error_deg{max_deg}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def print_results_table(results: Dict[int, Dict[str, Any]], func_name: str) -> None:
    """Pretty-print approximation error comparison table."""
    print(f"\n{'='*85}")
    print(f"  {func_name} — Approximation Error Comparison")
    print(f"{'='*85}")
    print(f"{'Deg':>4} {'Depth':>5} | {'Taylor L∞':>12} {'Cheby L∞':>12} {'LS L∞':>12} {'WM L∞':>12} {'WM wL∞':>12}")
    print(f"{'-'*4} {'-'*5}-+-{'-'*12}-{'-'*12}-{'-'*12}-{'-'*12}-{'-'*12}")
    for deg in sorted(results):
        row = results[deg]
        depth = row["depth"]
        t = row.get("taylor", {}).get("linf", float("inf"))
        c = row.get("chebyshev", {}).get("linf", float("inf"))
        l = row.get("least_squares", {}).get("linf", float("inf"))
        w = row.get("weighted_minimax", {}).get("linf", float("inf"))
        ww = row.get("weighted_minimax", {}).get("weighted_linf", float("inf"))
        print(f"{deg:4d} {depth:5d} | {t:12.6f} {c:12.6f} {l:12.6f} {w:12.6f} {ww:12.6f}")
