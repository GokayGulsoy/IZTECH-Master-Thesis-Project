"""Ext 4 — Cross-architecture backbone introspection.

Locates the encoder block list and the embedding module on a HuggingFace
sequence-classification model, regardless of whether it's BERT, RoBERTa,
or DistilBERT. Used by ``replace_activations``, ``linear_mixing``,
``quad_attention``, and ``hybrid_attention`` so they can operate on any
of the three architectures without duplicating layer-walking code.

Adding support for a new HF architecture is a one-line addition to the
``_BACKBONE_PATHS`` tuple below.
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn


# Each entry: (backbone-attr-name, layer-list-attribute-path).
# The path is a list of attribute names from the backbone to the
# nn.ModuleList of encoder blocks.
_BACKBONE_PATHS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("bert",       ("encoder", "layer")),
    ("roberta",    ("encoder", "layer")),
    ("distilbert", ("transformer", "layer")),
)


def _resolve_chain(obj: nn.Module, chain: Tuple[str, ...]) -> nn.Module:
    for attr in chain:
        obj = getattr(obj, attr)
    return obj


def get_backbone(model: nn.Module) -> Tuple[str, nn.Module]:
    """Return ``(backbone_name, backbone_module)``.

    ``backbone_name`` is one of ``{'bert', 'roberta', 'distilbert'}``.
    Raises ``AttributeError`` if no known backbone is found.
    """
    for name, _ in _BACKBONE_PATHS:
        bb = getattr(model, name, None)
        if bb is not None:
            return name, bb
    raise AttributeError(
        f"Model {type(model).__name__} exposes none of the supported "
        f"backbones {[n for n, _ in _BACKBONE_PATHS]}"
    )


def get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ``nn.ModuleList`` of encoder blocks for ``model``."""
    name, bb = get_backbone(model)
    for nm, chain in _BACKBONE_PATHS:
        if nm == name:
            return _resolve_chain(bb, chain)
    raise AssertionError("unreachable")  # pragma: no cover


def get_embeddings(model: nn.Module) -> nn.Module:
    """Return the embedding module of ``model``'s backbone."""
    _, bb = get_backbone(model)
    return bb.embeddings


def num_layers(model: nn.Module) -> int:
    return len(get_encoder_layers(model))
