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
import torch.nn.functional as F

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
        if t.ndim == 1:
            # Replicate [degree+1] → [num_heads, degree+1]
            t_expanded = t.unsqueeze(0).expand(num_heads, -1).clone()
        elif t.ndim == 2:
            if t.shape[0] != num_heads:
                raise ValueError(
                    f"Expected {num_heads} per-head coefficient rows, got {t.shape[0]}"
                )
            t_expanded = t.clone()
        else:
            raise ValueError(
                "cheb_coeffs must be 1D [degree+1] or 2D [num_heads, degree+1]"
            )
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


class SynthesizerAttention(nn.Module):
    """Self-attention without Q/K projections, using learned per-head patterns.

    The module keeps the original value projection and replaces the usual
    score path ``Q K^T / sqrt(d)`` with learned logits ``A~`` per head. During
    training, ``softmax(A~ + mask)`` is distilled against a teacher attention
    distribution; at export time, the resulting plaintext pattern is frozen and
    used by the FHE benchmark/inference path.
    """

    def __init__(
        self,
        value: nn.Linear,
        dropout: nn.Module,
        num_attention_heads: int,
        attention_head_size: int,
        max_seq_len: int,
        init_attention: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.value = nn.Linear(value.in_features, value.out_features, bias=value.bias is not None)
        self.value.load_state_dict(value.state_dict())
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = num_attention_heads * attention_head_size
        self.max_seq_len = max_seq_len

        init_logits = torch.zeros(
            num_attention_heads,
            max_seq_len,
            max_seq_len,
            dtype=value.weight.dtype,
        )
        if init_attention is not None:
            init_t = torch.as_tensor(init_attention, dtype=value.weight.dtype)
            if init_t.ndim != 3:
                raise ValueError(
                    "init_attention must have shape [num_heads, seq_len, seq_len]"
                )
            if init_t.shape[0] != num_attention_heads:
                raise ValueError(
                    f"Expected {num_attention_heads} heads, got {init_t.shape[0]}"
                )
            seq_len = int(init_t.shape[-1])
            if init_t.shape[1] != seq_len or seq_len > max_seq_len:
                raise ValueError(
                    f"init_attention must be square and <= max_seq_len={max_seq_len}"
                )
            probs = init_t.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            init_logits[:, :seq_len, :seq_len] = probs.log()
        self.pattern_logits = nn.Parameter(init_logits)

    @classmethod
    def from_self_attention(
        cls,
        self_attention: nn.Module,
        *,
        max_seq_len: int,
        init_attention: torch.Tensor | np.ndarray | None = None,
    ) -> "SynthesizerAttention":
        return cls(
            value=self_attention.value,
            dropout=self_attention.dropout,
            num_attention_heads=self_attention.num_attention_heads,
            attention_head_size=self_attention.attention_head_size,
            max_seq_len=max_seq_len,
            init_attention=init_attention,
        )

    def attention_pattern(self, seq_len: int | None = None) -> torch.Tensor:
        use_len = seq_len if seq_len is not None else self.max_seq_len
        logits = self.pattern_logits[:, :use_len, :use_len]
        return F.softmax(logits.float(), dim=-1).to(self.pattern_logits.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, ...] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError(
                "SynthesizerAttention currently supports encoder self-attention only"
            )
        if past_key_value is not None:
            raise NotImplementedError(
                "SynthesizerAttention does not support decoder key/value caches"
            )

        batch_size, seq_len, _ = hidden_states.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        value = self.value(hidden_states)
        value = value.view(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        attn_scores = self.pattern_logits[:, :seq_len, :seq_len]
        attn_scores = attn_scores.unsqueeze(0).expand(batch_size, -1, -1, -1)
        attn_scores = attn_scores.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(hidden_states.dtype)
        attn_probs = self.dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (context, attn_probs) if output_attentions else (context,)
