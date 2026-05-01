"""Quadratic-attention replacement for BERT self-attention.

Replaces the depth-heavy softmax-based attention with MPCFormer's
"2Quad" attention:

    Standard:  softmax(Q·K^T / sqrt(d)) · V
    2Quad:     ((Q·K^T)^2 / L) · V          (no softmax)

Properties under FHE/CKKS:
    - Eliminates polynomial-softmax (typically depth 4-5 with degree 12+)
    - Single ct×ct multiplication (Q·K^T) plus a squaring (also ct×ct)
    - No exp, no division, no normalization beyond scalar /L
    - Depth budget per layer drops from ~9 (LPAN) to ~5 (2Quad)

Precedent:
    - MPCFormer (Li et al., ICLR 2023) — original 2Quad design
    - MCPFormer / AutoFHE — refined polynomial attention variants
    - NEXUS (Yang et al., NDSS 2024) — uses similar low-degree polynomial replacements

Compared to LinearMixingAttention:
    - 2Quad keeps Q,K,V projections (preserves content-dependence)
    - Has more representational capacity than pure linear position mixing
    - Costs ~2× more under FHE but recovers ~2-3% accuracy in deep layers

Usage::

    from fhe_thesis.models.quad_attention import replace_attention_with_quad

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    replace_attention_with_quad(model, layer_indices=[4, 5, 6, 7])
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class QuadAttention(nn.Module):
    """Drop-in replacement for BertSelfAttention + BertSelfOutput.

    Computes per-head:
        scores  = (Q·K^T) / sqrt(d)
        weights = scores^2 / L                # quadratic, no softmax
        ctx     = weights · V
        out     = LN(W_o · concat(ctx_h) + b + x)

    This is the "2Quad" variant from MPCFormer.  All polynomial — no exp,
    no softmax, no normalization beyond a fixed scalar division.  Under FHE
    it costs 1 ct×ct (QK^T) + 1 ct×ct (squaring) + 1 ct×ct (·V) per head
    in terms of multiplicative depth, vs LPAN's 1 ct×ct + poly-softmax depth.

    Initialization:
        - Q, K, V projections: copied from original BertSelfAttention if
          available (`init_from_original=True`), else identity-initialized.
        - W_o: identity from BertSelfOutput.dense if available, else identity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layer_norm: nn.Module,
        original_attention: nn.Module | None = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projections (per-token plaintext × ciphertext)
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Post-attention LayerNorm (typically polynomial after LPAN Stage 3)
        self.LayerNorm = layer_norm
        self.dropout = nn.Dropout(attention_dropout)

        # Initialize from original BertSelfAttention if provided, else identity
        if original_attention is not None:
            self._copy_from_bert_attention(original_attention)
        else:
            for proj in (self.query, self.key, self.value, self.out_proj):
                nn.init.eye_(proj.weight)
                nn.init.zeros_(proj.bias)

    def _copy_from_bert_attention(self, orig: nn.Module) -> None:
        """Copy Q,K,V,W_o from a BertAttention module (BertSelfAttention + BertSelfOutput)."""
        # BertAttention has .self (BertSelfAttention) and .output (BertSelfOutput)
        self_attn = orig.self
        out_module = orig.output

        with torch.no_grad():
            self.query.weight.copy_(self_attn.query.weight)
            self.query.bias.copy_(self_attn.query.bias)
            self.key.weight.copy_(self_attn.key.weight)
            self.key.bias.copy_(self_attn.key.bias)
            self.value.weight.copy_(self_attn.value.weight)
            self.value.bias.copy_(self_attn.value.bias)
            self.out_proj.weight.copy_(out_module.dense.weight)
            self.out_proj.bias.copy_(out_module.dense.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, d)"""
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, d) -> (B, L, D)"""
        B, H, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        hidden_states: (B, L, D)
        attention_mask: (B, 1, 1, L) — additive mask (0 for keep, large negative for pad)

        Returns: (output,) or (output, weights) if output_attentions
        """
        B, L, D = hidden_states.shape

        # Project to Q, K, V
        Q = self._split_heads(self.query(hidden_states))  # (B, H, L, d)
        K = self._split_heads(self.key(hidden_states))    # (B, H, L, d)
        V = self._split_heads(self.value(hidden_states))  # (B, H, L, d)

        # Scaled dot-product scores: (B, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 2Quad: square scores + normalize by L (token count)
        # NOTE: no softmax — pure polynomial of degree 2 in scores
        weights = (scores * scores) / float(L)

        # Apply attention mask if provided (mask out padding by zeroing weights)
        # Standard additive mask uses large negatives → softmax→0; here we
        # zero out via multiplicative mask derived from the additive mask.
        if attention_mask is not None:
            # Convert additive mask (0 keep, -inf pad) to multiplicative (1 keep, 0 pad)
            keep = (attention_mask >= -1.0).to(weights.dtype)  # (B, 1, 1, L)
            weights = weights * keep

        weights = self.dropout(weights)

        # Weighted sum: (B, H, L, L) @ (B, H, L, d) -> (B, H, L, d)
        ctx = torch.matmul(weights, V)
        ctx = self._merge_heads(ctx)  # (B, L, D)

        # Output projection + residual + LN
        out = self.out_proj(ctx)
        out = self.LayerNorm(out + hidden_states)

        if output_attentions:
            return (out, weights)
        return (out,)


def replace_attention_with_quad(
    model: nn.Module,
    layer_indices: list[int] | None = None,
    num_heads: int | None = None,
    init_from_original: bool = True,
) -> nn.Module:
    """Replace BERT attention blocks with QuadAttention.

    Parameters
    ----------
    model : nn.Module
        HuggingFace BertForSequenceClassification (possibly LPAN'd).
    layer_indices : list[int] or None
        Layers to replace. None → all layers.
    num_heads : int or None
        Defaults to config.num_attention_heads.
    init_from_original : bool
        If True (default), copy Q,K,V,W_o from existing BertAttention.
        Crucial for warm-start fine-tuning.

    Returns
    -------
    Modified model (in-place + returned).
    """
    hidden_size = model.config.hidden_size
    if num_heads is None:
        num_heads = model.config.num_attention_heads
    allowed = set(layer_indices) if layer_indices is not None else None

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if allowed is not None and layer_idx not in allowed:
            continue

        # Skip if already a QuadAttention or LinearMixing replacement
        if isinstance(layer.attention, QuadAttention):
            continue

        # Grab the post-attention LayerNorm (original or polynomial)
        # If layer.attention is a standard BertAttention: layer.attention.output.LayerNorm
        # If already a custom replacement: layer.attention.LayerNorm
        if hasattr(layer.attention, "output") and hasattr(layer.attention.output, "LayerNorm"):
            post_attn_ln = layer.attention.output.LayerNorm
            original = layer.attention if init_from_original else None
        else:
            post_attn_ln = layer.attention.LayerNorm
            original = None  # custom replacement, no Q/K/V to copy

        quad = QuadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_norm=post_attn_ln,
            original_attention=original,
        )

        layer.attention = quad

    return model
