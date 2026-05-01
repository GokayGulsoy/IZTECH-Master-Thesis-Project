"""Hybrid attention model factory for HyPER-LPAN architecture.

Combines three layer types under one BERT model:
    - LinearMixingAttention: cheapest, for early layers (L0-L3)
    - QuadAttention (2Quad): middle layers (L4-L7), keeps Q/K/V no softmax
    - LPAN polynomial-softmax attention: deep layers (L8-L11), original LPAN

This module orchestrates per-layer replacements and provides the freeze
logic for progressive training across the three regimes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear_mixing import (
    MultiHeadLinearMixingAttention,
    replace_attention_with_linear_mixing,
)
from .quad_attention import QuadAttention, replace_attention_with_quad


def apply_hybrid_attention(
    model: nn.Module,
    linear_mixing_layers: list[int] | None = None,
    quad_attention_layers: list[int] | None = None,
    max_seq_len: int = 64,
    num_heads: int | None = None,
    init_quad_from_original: bool = True,
) -> nn.Module:
    """Apply hybrid attention replacement to a BERT model.

    Layers NOT in either list keep their existing attention (typically LPAN
    polynomial-softmax attention if applied beforehand).

    Parameters
    ----------
    model : nn.Module
        BertForSequenceClassification (typically with LPAN already applied).
    linear_mixing_layers : list[int] or None
        Layer indices to replace with MultiHeadLinearMixingAttention.
    quad_attention_layers : list[int] or None
        Layer indices to replace with QuadAttention (2Quad).
    max_seq_len : int
        Required for linear mixing layers.
    num_heads : int or None
        Defaults to config.num_attention_heads.
    init_quad_from_original : bool
        If True, copy Q,K,V,W_o weights from original BertAttention into Quad.

    Returns
    -------
    Modified model (in-place + returned).
    """
    linear_mixing_layers = linear_mixing_layers or []
    quad_attention_layers = quad_attention_layers or []

    overlap = set(linear_mixing_layers) & set(quad_attention_layers)
    if overlap:
        raise ValueError(f"Layers {overlap} cannot be both linear-mixing and quad")

    if quad_attention_layers:
        replace_attention_with_quad(
            model,
            layer_indices=quad_attention_layers,
            num_heads=num_heads,
            init_from_original=init_quad_from_original,
        )

    if linear_mixing_layers:
        replace_attention_with_linear_mixing(
            model,
            max_seq_len=max_seq_len,
            layer_indices=linear_mixing_layers,
            num_heads=num_heads,
        )

    return model


def get_layer_attention_type(layer_attention: nn.Module) -> str:
    """Return one of: 'linear_mixing', 'quad', 'lpan' (or original)."""
    if isinstance(layer_attention, MultiHeadLinearMixingAttention):
        return "linear_mixing"
    if isinstance(layer_attention, QuadAttention):
        return "quad"
    return "lpan"


def freeze_for_progressive_hybrid(
    model: nn.Module,
    replaced_layers: list[int],
    unfreeze_all_replaced_ffns: bool = True,
) -> int:
    """Freeze for progressive hybrid replacement (LPAN-style co-adaptation).

    Unfreezes:
    - All replaced layers' attention module params (Q,K,V,out_proj for Quad;
      pos_mix,out_proj for LinearMixing).
    - All replaced layers' FFN params (intermediate + output) if
      unfreeze_all_replaced_ffns=True (LPAN-style co-adaptation).
    - Classifier + pooler.

    Returns the number of trainable parameters.
    """
    replaced_set = set(replaced_layers)
    trainable = 0

    for name, param in model.named_parameters():
        should_train = False

        if name.startswith("classifier.") or name.startswith("bert.pooler."):
            should_train = True
        elif name.startswith("bert.encoder.layer."):
            parts = name.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])

            if li in replaced_set:
                # Replaced layers: unfreeze attention (mixing/quad)
                if rest.startswith("attention."):
                    should_train = True
                # And FFN if requested (co-adaptation)
                if unfreeze_all_replaced_ffns and (
                    rest.startswith("intermediate.") or rest.startswith("output.")
                ):
                    should_train = True

        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()

    return trainable


def freeze_for_global_finetune(model: nn.Module) -> int:
    """Unfreeze entire encoder + classifier for final global fine-tune."""
    trainable = 0
    for name, param in model.named_parameters():
        should_train = (
            name.startswith("bert.encoder.")
            or name.startswith("classifier.")
            or name.startswith("bert.pooler.")
        )
        param.requires_grad = should_train
        if should_train:
            trainable += param.numel()
    return trainable


def summarize_attention_types(model: nn.Module) -> dict[str, list[int]]:
    """Return dict: {'linear_mixing': [...], 'quad': [...], 'lpan': [...]}."""
    from fhe_thesis.models.backbone import get_encoder_layers
    summary = {"linear_mixing": [], "quad": [], "lpan": []}
    for li, layer in enumerate(get_encoder_layers(model)):
        kind = get_layer_attention_type(layer.attention)
        summary[kind].append(li)
    return summary
