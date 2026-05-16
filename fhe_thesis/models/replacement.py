"""Model surgery: replace BERT activations with polynomial modules.

Unified from finetune_bert_tiny.py (2-arg) and multi_model_eval.py (3-arg).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn

from ..config import MAX_INTERVALS
from .activations import (
    PerHeadPolynomialSoftmax,
    PolynomialGELU,
    PolynomialLayerNorm,
    PolynomialSoftmax,
    SynthesizerAttention,
)


def _get_attention_self_module(layer: nn.Module) -> nn.Module:
    attn = getattr(layer, "attention", None)
    if attn is None or getattr(attn, "self", None) is None:
        raise TypeError(
            f"Layer type {type(layer).__name__} does not expose layer.attention.self"
        )
    return attn.self


_LAYER_INDEX_RE = re.compile(r"(?:encoder|transformer)\.layer\.(\d+)\.")


def _parse_layer_index(name: str) -> Optional[int]:
    match = _LAYER_INDEX_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_interval(
    module_name: str,
    op_name: str,
    interval_overrides: Optional[Dict[str, Sequence[float]]] = None,
) -> tuple[float, float]:
    if interval_overrides is not None and module_name in interval_overrides:
        lo, hi = interval_overrides[module_name]
        return float(lo), float(hi)
    lo, hi = MAX_INTERVALS[op_name]
    return float(lo), float(hi)


def build_poly_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    interval_overrides: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Reconstruct replacement config from a saved LPAN state dict.

    This is primarily used to rebuild Stage-3 checkpoints before applying the
    Stage-4 Synthesizer attention swap.  If explicit interval metadata is not
    available, the function falls back to the global safe maxima in
    ``config.MAX_INTERVALS`` so legacy checkpoints remain loadable.
    """
    poly_coeffs: Dict[str, Dict[str, Any]] = {}

    for name, tensor in state_dict.items():
        layer_idx = _parse_layer_index(name)
        if layer_idx is None or not name.endswith(".coeffs"):
            continue

        coeffs = tensor.detach().cpu().numpy().astype(np.float32)
        module_name = name[: -len(".coeffs")]

        if module_name.endswith("intermediate.intermediate_act_fn"):
            poly_coeffs[f"L{layer_idx}_GELU"] = {
                "cheb_coeffs": coeffs,
                "interval": _resolve_interval(module_name, "GELU", interval_overrides),
                "degree": int(coeffs.shape[-1] - 1),
            }
        elif module_name.endswith("attention.self.poly_softmax"):
            poly_coeffs[f"L{layer_idx}_Softmax"] = {
                "cheb_coeffs": coeffs,
                "interval": _resolve_interval(module_name, "Softmax", interval_overrides),
                "degree": int(coeffs.shape[-1] - 1),
            }
        elif module_name.endswith("attention.output.LayerNorm"):
            poly_coeffs[f"L{layer_idx}_LN_attn"] = {
                "cheb_coeffs": coeffs,
                "interval": _resolve_interval(module_name, "LN", interval_overrides),
                "degree": int(coeffs.shape[-1] - 1),
            }
        elif module_name.endswith("output.LayerNorm"):
            poly_coeffs[f"L{layer_idx}_LN_out"] = {
                "cheb_coeffs": coeffs,
                "interval": _resolve_interval(module_name, "LN", interval_overrides),
                "degree": int(coeffs.shape[-1] - 1),
            }

    return poly_coeffs


def replace_attention_with_synthesizer(
    model: nn.Module,
    *,
    max_seq_len: int = 128,
    layer_indices: Optional[Sequence[int]] = None,
    init_patterns: Optional[Dict[int, torch.Tensor]] = None,
) -> nn.Module:
    """Replace selected self-attention blocks with SynthesizerAttention.

    ``init_patterns`` is an optional mapping ``layer_idx -> [H, S, S]`` used to
    seed the learned attention logits from teacher averages.
    """
    from fhe_thesis.models.backbone import get_encoder_layers

    allowed_layers: Optional[Set[int]] = (
        set(layer_indices) if layer_indices is not None else None
    )
    for layer_idx, layer in enumerate(get_encoder_layers(model)):
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        self_attn = _get_attention_self_module(layer)
        init_attention = None if init_patterns is None else init_patterns.get(layer_idx)
        layer.attention.self = SynthesizerAttention.from_self_attention(
            self_attn,
            max_seq_len=max_seq_len,
            init_attention=init_attention,
        )
    return model


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
        Optionally for LayerNorm: 'L{i}_LN_attn' and 'L{i}_LN_out'.
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

    from fhe_thesis.models.backbone import get_encoder_layers
    for layer_idx, layer in enumerate(get_encoder_layers(model)):
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        # 1. Replace GELU
        gelu_key = f"L{layer_idx}_GELU"
        if "GELU" in types and gelu_key in poly_coeffs:
            pc = poly_coeffs[gelu_key]
            layer.intermediate.intermediate_act_fn = PolynomialGELU(
                pc["cheb_coeffs"], pc["interval"], learnable=learnable
            )

        # 2. Replace Softmax — preserve shared-vs-per-head coeff layout
        sm_key = f"L{layer_idx}_Softmax"
        if "Softmax" in types and sm_key in poly_coeffs:
            pc = poly_coeffs[sm_key]
            attn = _get_attention_self_module(layer)
            num_heads = attn.num_attention_heads
            coeffs = np.asarray(pc["cheb_coeffs"])
            if coeffs.ndim == 1:
                poly_sm = PolynomialSoftmax(
                    coeffs, pc["interval"], learnable=learnable
                )
            else:
                poly_sm = PerHeadPolynomialSoftmax(
                    num_heads, coeffs, pc["interval"], learnable=learnable
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
        attn_ln_key = f"L{layer_idx}_LN_attn"
        out_ln_key = f"L{layer_idx}_LN_out"
        if "LN" in types and (
            ln_key in poly_coeffs or attn_ln_key in poly_coeffs or out_ln_key in poly_coeffs
        ):
            attn_pc = poly_coeffs.get(attn_ln_key, poly_coeffs.get(ln_key))
            out_pc = poly_coeffs.get(out_ln_key, poly_coeffs.get(ln_key))
            if attn_pc is None or out_pc is None:
                raise KeyError(
                    f"Layer {layer_idx} is missing LayerNorm polynomial metadata"
                )
            layer.attention.output.LayerNorm = PolynomialLayerNorm(
                hidden_size, layer.attention.output.LayerNorm,
                attn_pc["cheb_coeffs"], attn_pc["interval"], learnable=learnable,
            )
            layer.output.LayerNorm = PolynomialLayerNorm(
                hidden_size, layer.output.LayerNorm,
                out_pc["cheb_coeffs"], out_pc["interval"], learnable=learnable,
            )

    return model
