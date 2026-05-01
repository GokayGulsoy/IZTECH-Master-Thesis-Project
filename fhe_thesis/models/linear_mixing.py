"""Linear mixing replacement for BERT self-attention.

Replaces the O(L²) ciphertext×ciphertext Q·K^T·V attention mechanism with
two learned plaintext linear layers:

    P ∈ R^{L×L}    — position mixing (cross-token, replaces Q·K^T softmax)
    W_mix ∈ R^{d×d} — feature mixing (per-token, replaces V + O projection)

Under FHE this is entirely plaintext × ciphertext operations — zero ct×ct
multiplications — reducing attention cost by ~400× per layer while preserving
the PF-SR (Pure-FHE Single-Round) protocol guarantee.

Precedent: MPCFormer-Quad, FNet, gMLP all replace attention with linear/fixed
cross-token mixing and report ≤2% accuracy drop on SST-2.

Usage::

    from fhe_thesis.models.linear_mixing import replace_attention_with_linear_mixing

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    replace_attention_with_linear_mixing(model, max_seq_len=64)
    # Fine-tune only mixing layers + classifier
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadLinearMixingAttention(nn.Module):
    """Multi-head drop-in replacement for BertSelfAttention + BertSelfOutput.

    Each of `num_heads` heads has its own position-mixing matrix P_h operating
    on a head_dim-sized slice of the hidden dimension.  This mirrors multi-head
    attention's structure (each head learns a different mixing pattern) while
    keeping ALL operations as plaintext × ciphertext — zero ct×ct.

    Computes:
        For head h:  z_h = P_h @ x_h          (position mixing per head)
        Concat:      z   = [z_0; z_1; ...; z_H]
        Output:      output = LN(W_out @ z + b + x)   (output projection + residual)

    Initialization: all P_h and W_out initialized to identity for pass-through.
    """

    def __init__(self, hidden_size: int, max_seq_len: int,
                 layer_norm: nn.Module, num_heads: int = 12) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Per-head position mixing: each head gets its own (L, L) matrix
        # Stored as a single parameter for efficiency: (num_heads, L, L)
        self.pos_mix_weight = nn.Parameter(
            torch.eye(max_seq_len).unsqueeze(0).expand(num_heads, -1, -1).clone()
        )
        self.pos_mix_bias = nn.Parameter(torch.zeros(num_heads, max_seq_len))

        # Output projection: (hidden, hidden) — combines head outputs
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Post-attention LayerNorm (kept from original or polynomial)
        self.LayerNorm = layer_norm

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
        hidden_states: (batch, seq_len, hidden)
        Returns tuple: (output,) or (output, None) if output_attentions
        """
        B, L, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        # Split into heads: (B, L, H, d) -> (B, H, L, d)
        x_heads = hidden_states.view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)

        # Position mixing per head: (B, H, d, L) @ P_h^T -> (B, H, d, L)
        # Transpose to (B, H, d, L) for matrix multiply
        x_t = x_heads.transpose(2, 3)  # (B, H, d, L)

        # P_h: (H, L_max, L_max) — slice to actual seq len
        P = self.pos_mix_weight[:, :L, :L]  # (H, L, L)
        bias = self.pos_mix_bias[:, :L]      # (H, L)

        # Batched matmul: (B, H, d, L) @ (H, L, L)^T -> (B, H, d, L)
        # Expand P for batch: (1, H, L, L) -> broadcast with (B, H, d, L)
        mixed = torch.matmul(x_t, P.transpose(-2, -1))  # (B, H, d, L)
        mixed = mixed + bias.unsqueeze(0).unsqueeze(2)    # broadcast bias

        # Transpose back: (B, H, d, L) -> (B, H, L, d) -> (B, L, H, d) -> (B, L, D)
        mixed = mixed.transpose(2, 3)  # (B, H, L, d)
        mixed = mixed.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # Output projection: combines all heads
        mixed = self.out_proj(mixed)  # (B, L, D)

        # Residual + LayerNorm
        output = self.LayerNorm(mixed + hidden_states)

        if output_attentions:
            return (output, None)
        return (output,)


# Keep backward compatibility alias
LinearMixingAttention = MultiHeadLinearMixingAttention


def replace_attention_with_linear_mixing(
    model: nn.Module,
    max_seq_len: int = 64,
    layer_indices: list[int] | None = None,
    num_heads: int | None = None,
) -> nn.Module:
    """Replace BERT attention blocks with MultiHeadLinearMixingAttention.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace BertForSequenceClassification model (possibly with
        LPAN polynomial activations already applied).
    max_seq_len : int
        Maximum sequence length for the position mixing matrix.
    layer_indices : list[int] or None
        If provided, only replace attention in these layers. If None,
        replace all layers.
    num_heads : int or None
        Number of mixing heads. Defaults to model's config.num_attention_heads.

    Returns
    -------
    nn.Module
        Modified model (in-place + returned).
    """
    hidden_size = model.config.hidden_size
    if num_heads is None:
        num_heads = model.config.num_attention_heads
    allowed = set(layer_indices) if layer_indices is not None else None

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if allowed is not None and layer_idx not in allowed:
            continue

        # Skip if already replaced
        if isinstance(layer.attention, MultiHeadLinearMixingAttention):
            continue

        # Grab the post-attention LayerNorm (original or polynomial)
        post_attn_ln = layer.attention.output.LayerNorm

        # Create multi-head linear mixing replacement
        mixing = MultiHeadLinearMixingAttention(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            layer_norm=post_attn_ln,
            num_heads=num_heads,
        )

        # Replace the entire attention block
        layer.attention = mixing

    return model


def freeze_for_mixing_finetune(model: nn.Module) -> int:
    """Freeze everything except mixing layers + classifier head.

    Returns the number of trainable parameters.
    """
    trainable = 0
    for name, param in model.named_parameters():
        should_train = (
            "pos_mix" in name
            or "out_proj" in name
            or "feat_mix" in name  # backward compat with old single-head
            or name.startswith("classifier.")
            or name.startswith("bert.pooler.")
        )
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
    return trainable


def freeze_for_progressive_mixing(
    model: nn.Module,
    current_layer: int,
    replaced_layers: list[int],
) -> int:
    """Freeze for progressive layer-by-layer mixing replacement.

    Unfreezes:
    - All replaced layers' mixing module params (pos_mix, feat_mix, LN)
      so previously replaced layers co-adapt with the new one.
    - Current layer's FFN (intermediate.dense + output.dense + output.LN)
      so it can adapt to the new mixing output distribution.
    - Classifier + pooler.

    Everything else is frozen.

    Returns the number of trainable parameters.
    """
    replaced_set = set(replaced_layers)
    trainable = 0

    for name, param in model.named_parameters():
        should_train = False

        # Classifier + pooler
        if name.startswith("classifier.") or name.startswith("bert.pooler."):
            should_train = True
        elif name.startswith("bert.encoder.layer."):
            parts = name.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])

            # All params in replaced layers' attention module (= mixing module)
            if li in replaced_set and rest.startswith("attention."):
                should_train = True

            # Current layer's FFN: intermediate + output (NOT attention)
            if li == current_layer:
                if rest.startswith("intermediate.") or rest.startswith("output."):
                    should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()

    return trainable
