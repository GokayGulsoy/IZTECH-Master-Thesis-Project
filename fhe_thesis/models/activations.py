"""Polynomial activation modules: PolynomialGELU, PolynomialSoftmax, PolynomialLayerNorm.

Supports two modes:
  - Fixed coefficients (register_buffer): original Chebyshev approximation.
  - Learnable coefficients (nn.Parameter): LPAN — coefficients are optimized
    jointly with model weights during fine-tuning.

Uses hard clamping with tight, realistic bounds to prevent NaN cascades.
Polynomial evaluation is forced to fp32 to avoid fp16 overflow in Clenshaw recurrence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..poly.chebyshev import cheb_eval_torch, cheb_eval_per_head_torch
from ..config import Interval


def _cheb_eval_fp32(coeffs: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Evaluate Chebyshev polynomial in fp32 for numerical stability.

    Polynomial evaluation with Clenshaw recurrence is sensitive to
    precision — fp16 intermediate values can overflow for degree ≥ 6.
    """
    return cheb_eval_torch(coeffs, xs.float()).to(xs.dtype)


def _cheb_eval_per_head_fp32(coeffs: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Per-head batched Chebyshev evaluation in fp32.

    coeffs : [num_heads, degree+1]
    xs     : [batch, num_heads, seq_len, seq_len]
    """
    return cheb_eval_per_head_torch(coeffs.float(), xs.float()).to(xs.dtype)


class PolynomialGELU(nn.Module):
    """Replace GELU with a Chebyshev polynomial approximation.

    Parameters
    ----------
    cheb_coeffs : np.ndarray
        Initial Chebyshev coefficients.
    interval : Interval
        (a, b) approximation domain.
    learnable : bool
        If True, coefficients are nn.Parameter (LPAN mode).
    """

    def __init__(self, cheb_coeffs: np.ndarray, interval: Interval,
                 learnable: bool = False) -> None:
        super().__init__()
        t = torch.tensor(cheb_coeffs, dtype=torch.float32)
        if learnable:
            self.coeffs = nn.Parameter(t)
        else:
            self.register_buffer("coeffs", t)
        self.a, self.b = interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.clamp(self.a, self.b)
        xs = (2.0 * xc - (self.a + self.b)) / (self.b - self.a)
        return _cheb_eval_fp32(self.coeffs, xs).clamp(-1e4, 1e4)


class PolynomialSoftmax(nn.Module):
    """Replace softmax's exp() with a polynomial approximation.

    Parameters
    ----------
    cheb_coeffs : np.ndarray
        Initial Chebyshev coefficients.
    interval : Interval
        (a, b) approximation domain.
    learnable : bool
        If True, coefficients are nn.Parameter (LPAN mode).
    """

    def __init__(self, cheb_coeffs: np.ndarray, interval: Interval,
                 learnable: bool = False) -> None:
        super().__init__()
        t = torch.tensor(cheb_coeffs, dtype=torch.float32)
        if learnable:
            self.coeffs = nn.Parameter(t)
        else:
            self.register_buffer("coeffs", t)
        self.a, self.b = interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shifted = x - x.max(dim=-1, keepdim=True).values
        xc = x_shifted.clamp(self.a, self.b)
        xs = (2.0 * xc - (self.a + self.b)) / (self.b - self.a)
        # exp() output is always positive; clamp to [eps, reasonable_max]
        exp_approx = _cheb_eval_fp32(self.coeffs, xs).clamp(min=1e-8, max=1e4)
        return exp_approx / exp_approx.sum(dim=-1, keepdim=True)


class PerHeadPolynomialSoftmax(nn.Module):
    """Per-head polynomial softmax: each attention head learns its own coefficients.

    Instead of one Chebyshev polynomial shared across all heads in a layer,
    this module maintains *num_heads* independent coefficient sets.  Each
    head can specialise its exp() approximation to its own score distribution
    (positional, semantic, syntactic), giving a much tighter fit than a
    one-size-fits-all polynomial.

    Adds num_heads × (degree+1) parameters per layer but keeps the same
    multiplicative depth as the shared version.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    cheb_coeffs : np.ndarray
        Initial Chebyshev coefficients [degree+1].  Replicated to all heads.
    interval : Interval
        (a, b) approximation domain.
    learnable : bool
        If True, coefficients are nn.Parameter (LPAN mode).
    """

    def __init__(self, num_heads: int, cheb_coeffs: np.ndarray,
                 interval: Interval, learnable: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.a, self.b = interval
        t = torch.tensor(cheb_coeffs, dtype=torch.float32)
        # Replicate [degree+1] → [num_heads, degree+1]
        t_expanded = t.unsqueeze(0).expand(num_heads, -1).clone()
        if learnable:
            self.coeffs = nn.Parameter(t_expanded)
        else:
            self.register_buffer("coeffs", t_expanded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_heads, seq_len, seq_len]"""
        x_shifted = x - x.max(dim=-1, keepdim=True).values
        xc = x_shifted.clamp(self.a, self.b)
        xs = (2.0 * xc - (self.a + self.b)) / (self.b - self.a)
        exp_approx = _cheb_eval_per_head_fp32(self.coeffs, xs).clamp(min=1e-8, max=1e4)
        return exp_approx / exp_approx.sum(dim=-1, keepdim=True)


class PolynomialLayerNorm(nn.Module):
    """Replace LayerNorm's 1/sqrt(var+eps) with a polynomial approximation.

    Parameters
    ----------
    normalized_shape : int
        Hidden dimension.
    orig_layer_norm : nn.LayerNorm
        Original LayerNorm (for weight/bias initialization).
    cheb_coeffs : np.ndarray
        Initial Chebyshev coefficients.
    interval : Interval
        (a, b) approximation domain.
    learnable : bool
        If True, coefficients are nn.Parameter (LPAN mode).
    """

    def __init__(
        self,
        normalized_shape: int,
        orig_layer_norm: nn.LayerNorm,
        cheb_coeffs: np.ndarray,
        interval: Interval,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(orig_layer_norm.weight.data.clone())
        self.bias = nn.Parameter(orig_layer_norm.bias.data.clone())
        self.eps = orig_layer_norm.eps
        t = torch.tensor(cheb_coeffs, dtype=torch.float32)
        if learnable:
            self.coeffs = nn.Parameter(t)
        else:
            self.register_buffer("coeffs", t)
        self.a, self.b = interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        var_eps = (var + self.eps).clamp(self.a, self.b)
        xs = (2.0 * var_eps - (self.a + self.b)) / (self.b - self.a)
        inv_std = _cheb_eval_fp32(self.coeffs, xs).clamp(min=1e-6, max=1e4)
        result = self.weight * ((x - mean) * inv_std) + self.bias
        return result.clamp(-1e4, 1e4)
