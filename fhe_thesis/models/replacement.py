"""Model surgery: replace BERT activations with polynomial modules.

Unified from finetune_bert_tiny.py (2-arg) and multi_model_eval.py (3-arg).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from .activations import PolynomialGELU, PolynomialSoftmax, PerHeadPolynomialSoftmax, PolynomialLayerNorm


def replace_activations(
    model: nn.Module,
    poly_coeffs: Dict[str, Any],
    hidden_size: Optional[int] = None,
    learnable: bool = False,
    replace_types: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """Replace BERT activations with polynomial versions.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace BERT model for sequence classification.
    poly_coeffs : dict
        Keys: 'L{i}_GELU', 'L{i}_Softmax', 'L{i}_LN'.
        Values: dict with 'cheb_coeffs', 'interval', 'degree'.
    hidden_size : int, optional
        Hidden dimension. If None, read from model.config.hidden_size.
    learnable : bool
        If True, polynomial coefficients are nn.Parameter (LPAN mode)
        and will be optimized during training.
    replace_types : list of str, optional
        Which activation types to replace. Subset of ["GELU", "Softmax", "LN"].
        If None, replaces all three types.
    layer_indices : list of int, optional
        Which layers to apply replacements to. If None, applies to all layers.

    Returns
    -------
    nn.Module
        The modified model (in-place mutation + returned).
    """
    if hidden_size is None:
        hidden_size = model.config.hidden_size

    types: Set[str] = set(replace_types) if replace_types else {"GELU", "Softmax", "LN"}
    allowed_layers: Optional[Set[int]] = set(layer_indices) if layer_indices is not None else None

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        # 1. Replace GELU
        gelu_key = f"L{layer_idx}_GELU"
        if "GELU" in types and gelu_key in poly_coeffs:
            pc = poly_coeffs[gelu_key]
            layer.intermediate.intermediate_act_fn = PolynomialGELU(
                pc["cheb_coeffs"], pc["interval"], learnable=learnable
            )

        # 2. Replace Softmax — per-head coefficients for better approximation
        sm_key = f"L{layer_idx}_Softmax"
        if "Softmax" in types and sm_key in poly_coeffs:
            pc = poly_coeffs[sm_key]
            attn = layer.attention.self
            num_heads = attn.num_attention_heads
            poly_sm = PerHeadPolynomialSoftmax(
                num_heads, pc["cheb_coeffs"], pc["interval"], learnable=learnable
            )
            # Register as a named submodule so state_dict includes coeffs
            attn.poly_softmax = poly_sm

            def make_patched(attn_mod, psm):
                def patched(hidden_states, attention_mask=None, head_mask=None,
                           encoder_hidden_states=None, encoder_attention_mask=None,
                           past_key_value=None, output_attentions=False):
                    Q = attn_mod.query(hidden_states)
                    K = attn_mod.key(hidden_states)
                    V = attn_mod.value(hidden_states)
                    bs = Q.size(0)
                    nh = attn_mod.num_attention_heads
                    hd = attn_mod.attention_head_size
                    Q = Q.view(bs, -1, nh, hd).transpose(1, 2)
                    K = K.view(bs, -1, nh, hd).transpose(1, 2)
                    V = V.view(bs, -1, nh, hd).transpose(1, 2)
                    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(hd)
                    if attention_mask is not None:
                        scores = scores + attention_mask
                    attn_probs = psm(scores)
                    if head_mask is not None:
                        attn_probs = attn_probs * head_mask
                    ctx = torch.matmul(attn_probs, V)
                    ctx = ctx.transpose(1, 2).contiguous().view(bs, -1, nh * hd)
                    return (ctx, attn_probs) if output_attentions else (ctx,)
                return patched

            attn.forward = make_patched(attn, poly_sm)

        # 3. Replace LayerNorms
        ln_key = f"L{layer_idx}_LN"
        if "LN" in types and ln_key in poly_coeffs:
            pc = poly_coeffs[ln_key]
            layer.attention.output.LayerNorm = PolynomialLayerNorm(
                hidden_size, layer.attention.output.LayerNorm,
                pc["cheb_coeffs"], pc["interval"], learnable=learnable,
            )
            layer.output.LayerNorm = PolynomialLayerNorm(
                hidden_size, layer.output.LayerNorm,
                pc["cheb_coeffs"], pc["interval"], learnable=learnable,
            )

    return model
