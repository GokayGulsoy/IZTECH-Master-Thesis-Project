"""Ext 2 — Learnable polynomial coefficients with range-adaptive distillation.

The fixed-coefficient Chebyshev/Remez approximations used by LPAN are
optimal *in the L^∞ sense over a fixed input range*. In practice the
true input distribution at each layer is much narrower than the worst-
case range used to derive coefficients, so the approximation wastes
its budget on tails that are never visited.

This module makes the approximation polynomials trainable jointly
with the model, with two additional ingredients:

1. **Range tracking**: each ``LearnablePolyAdapter`` maintains EMA
   running min/max of its inputs. The polynomial is evaluated on the
   normalised domain [-1, 1] mapped to the *empirical* range, not the
   theoretical worst case. This is updated only during training; at
   inference the frozen empirical range is used.

2. **Function-fidelity regulariser**: a small ``||p(x) − f(x)||^2``
   penalty over a Chebyshev-node grid keeps the learned polynomial
   close to the target activation (GELU, softmax, etc.) so the
   network cannot "cheat" by overfitting an arbitrary polynomial.

FHE compliance: at inference time, both the coefficients and the
empirical range are frozen plaintext constants. They are baked into
the published model (just like the original Chebyshev coefficients).
No ciphertext-dependent computation occurs at deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .chebyshev import cheb_eval_torch


@dataclass
class RangeStats:
    min: float
    max: float


class LearnablePolyAdapter(nn.Module):
    """Wraps a Chebyshev approximation as a trainable PyTorch module.

    Parameters
    ----------
    init_coeffs : torch.Tensor
        Initial Chebyshev series coefficients (shape ``(degree+1,)``).
    target_fn : callable
        The reference activation, e.g. ``torch.nn.functional.gelu``.
        Used by the fidelity regulariser. Must be torch-vectorisable.
    init_range : (float, float)
        Initial ``(min, max)`` of the polynomial's input domain.
    range_ema : float
        EMA decay for empirical range tracking (default 0.99).
    fidelity_grid_size : int
        Number of Chebyshev nodes for the fidelity regulariser.
    name : str
        Human-readable identifier (e.g. ``"L7.softmax"``) for logging.
    """

    def __init__(
        self,
        init_coeffs: torch.Tensor,
        target_fn: Callable[[torch.Tensor], torch.Tensor],
        init_range: tuple[float, float],
        range_ema: float = 0.99,
        fidelity_grid_size: int = 64,
        name: str = "poly",
    ) -> None:
        super().__init__()
        self.coeffs = nn.Parameter(init_coeffs.clone().detach().float())
        self.target_fn = target_fn
        # Range buffers (saved in state_dict, not optimised)
        self.register_buffer("range_min", torch.tensor(float(init_range[0])))
        self.register_buffer("range_max", torch.tensor(float(init_range[1])))
        self.range_ema = float(range_ema)
        self.name = name

        # Pre-compute Chebyshev nodes on [-1, 1] for the fidelity loss.
        k = torch.arange(fidelity_grid_size).float()
        nodes = torch.cos(torch.pi * (k + 0.5) / fidelity_grid_size)
        self.register_buffer("fidelity_nodes", nodes)

    # ---- forward ----------------------------------------------------
    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` from current empirical range to [-1, 1]."""
        a = self.range_min
        b = self.range_max
        scale = 0.5 * (b - a) + 1e-12
        return (x - 0.5 * (a + b)) / scale

    def _denormalise(self, u: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_normalise``."""
        a = self.range_min
        b = self.range_max
        return 0.5 * (a + b) + 0.5 * (b - a) * u

    def update_range(self, x: torch.Tensor) -> None:
        """EMA-update the empirical input range from a training batch."""
        with torch.no_grad():
            x_min = float(x.min().item())
            x_max = float(x.max().item())
            self.range_min.mul_(self.range_ema).add_(
                (1.0 - self.range_ema) * x_min)
            self.range_max.mul_(self.range_ema).add_(
                (1.0 - self.range_ema) * x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update_range(x)
        u = self._normalise(x).clamp(-1.0, 1.0)
        return cheb_eval_torch(self.coeffs, u)

    # ---- regulariser ------------------------------------------------
    def fidelity_loss(self) -> torch.Tensor:
        """Return ``mean (p(x) − f(x))^2`` on Chebyshev nodes."""
        u = self.fidelity_nodes
        x = self._denormalise(u)
        with torch.no_grad():
            target = self.target_fn(x)
        pred = cheb_eval_torch(self.coeffs, u)
        return ((pred - target) ** 2).mean()

    def export_coeffs(self) -> torch.Tensor:
        """Return the trained Chebyshev coefficients (CPU, detached)."""
        return self.coeffs.detach().cpu()

    def get_range(self) -> RangeStats:
        return RangeStats(min=float(self.range_min.item()),
                          max=float(self.range_max.item()))


def collect_fidelity_loss(model: nn.Module) -> torch.Tensor:
    """Sum ``fidelity_loss`` over every ``LearnablePolyAdapter`` in ``model``.

    Returns a scalar tensor (zero if no adapters are present).
    """
    losses = []
    for m in model.modules():
        if isinstance(m, LearnablePolyAdapter):
            losses.append(m.fidelity_loss())
    if not losses:
        # Return a zero on the model's device.
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(losses).sum()


def export_adapters_state(model: nn.Module) -> dict:
    """Return ``{name: {coeffs, range_min, range_max}}`` for export to FHE."""
    out: dict = {}
    for m in model.modules():
        if isinstance(m, LearnablePolyAdapter):
            r = m.get_range()
            out[m.name] = {
                "coeffs": m.export_coeffs().tolist(),
                "range_min": r.min,
                "range_max": r.max,
            }
    return out
