"""Chebyshev polynomial evaluation (Clenshaw recurrence) for PyTorch and NumPy."""

from __future__ import annotations

import numpy as np
import torch


def cheb_eval_torch(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a Chebyshev series on [-1, 1] using Clenshaw recurrence.

    Parameters
    ----------
    coeffs : torch.Tensor
        Chebyshev series coefficients [c0, c1, ..., cn].
    x : torch.Tensor
        Points in [-1, 1] at which to evaluate.

    Returns
    -------
    torch.Tensor
        Polynomial values, same shape as *x*.
    """
    if len(coeffs) == 1:
        return coeffs[0] * torch.ones_like(x)
    b_k2 = torch.zeros_like(x)
    b_k1 = torch.zeros_like(x)
    for k in range(len(coeffs) - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + x * b_k1 - b_k2


def cheb_eval_per_head_torch(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched Chebyshev evaluation with per-head coefficients (Clenshaw).

    Evaluates *num_heads* independent Chebyshev polynomials in a single
    vectorised pass.  The loop iterates over polynomial degree (typically
    8-12), NOT over heads — so this is fully parallelised across heads,
    batch size, and sequence length.

    Parameters
    ----------
    coeffs : torch.Tensor
        [num_heads, degree+1] — one coefficient set per head.
    x : torch.Tensor
        [batch, num_heads, seq_len, seq_len] — scaled inputs per head.

    Returns
    -------
    torch.Tensor
        Same shape as *x*.
    """
    if coeffs.shape[1] == 1:
        return coeffs[:, 0].view(1, -1, 1, 1) * torch.ones_like(x)
    b_k2 = torch.zeros_like(x)
    b_k1 = torch.zeros_like(x)
    for k in range(coeffs.shape[1] - 1, 0, -1):
        c_k = coeffs[:, k].view(1, -1, 1, 1)
        b_k = c_k + 2.0 * x * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k
    c_0 = coeffs[:, 0].view(1, -1, 1, 1)
    return c_0 + x * b_k1 - b_k2


def chebyshev_to_power(cheb_coeffs: np.ndarray) -> np.ndarray:
    """Convert Chebyshev series coefficients to standard power basis.

    Parameters
    ----------
    cheb_coeffs : np.ndarray
        Chebyshev coefficients [c0, c1, ..., cn] on [-1, 1].

    Returns
    -------
    np.ndarray
        Power-basis coefficients [a0, a1, ..., an] such that
        p(x) = a0 + a1*x + a2*x^2 + ...
    """
    from numpy.polynomial import chebyshev as cheb_mod
    return cheb_mod.cheb2poly(cheb_coeffs)
