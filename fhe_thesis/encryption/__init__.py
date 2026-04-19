"""CKKS encryption context and homomorphic operations.

Public surface for the LPAN-FHE protocol (see ``docs/ckks_protocol.md``).

Imports are deliberately lazy so that backend-agnostic modules
(``depth``, ``packing``) can be loaded on machines without TenSEAL.
"""

from __future__ import annotations

from .depth import DEPTH_COST, DepthAudit, transformer_layer_depth

__all__ = [
    "BackendCapabilities",
    "CKKSBackend",
    "DEPTH_COST",
    "DepthAudit",
    "PolyCoeffs",
    "TenSEALBackend",
    "TokenPackedTensor",
    "create_ckks_context",
    "enc_attention_apply",
    "enc_gelu_poly",
    "enc_linear",
    "enc_ln_poly",
    "enc_qk_scores",
    "enc_self_attention",
    "enc_softmax_poly",
    "encrypt_attention_block",
    "encrypt_ffn_block",
    "encrypt_inference",
    "encrypt_layer",
    "load_coefficients",
    "load_model_weights",
    "make_context",
    "run_phase",
    "transformer_layer_depth",
]


def __getattr__(name):  # PEP 562 lazy import
    if name in {"create_ckks_context", "make_context"}:
        from . import context

        return getattr(context, name)
    if name in {"BackendCapabilities", "CKKSBackend", "TenSEALBackend"}:
        from . import backend

        return getattr(backend, name)
    if name == "TokenPackedTensor":
        from . import packing

        return packing.TokenPackedTensor
    if name in {
        "enc_attention_apply",
        "enc_gelu_poly",
        "enc_linear",
        "enc_ln_poly",
        "enc_qk_scores",
        "enc_self_attention",
        "enc_softmax_poly",
    }:
        from . import ops

        return getattr(ops, name)
    if name in {"PolyCoeffs", "load_coefficients"}:
        from . import coefficients

        return getattr(coefficients, name)
    if name in {
        "encrypt_attention_block",
        "encrypt_ffn_block",
        "encrypt_inference",
        "encrypt_layer",
        "load_model_weights",
        "run_phase",
    }:
        from . import protocol

        return getattr(protocol, name)
    raise AttributeError(f"module 'fhe_thesis.encryption' has no attribute {name!r}")
