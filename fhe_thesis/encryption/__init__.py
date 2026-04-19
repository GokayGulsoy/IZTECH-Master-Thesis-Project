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
    "TenSEALBackend",
    "TokenPackedTensor",
    "create_ckks_context",
    "enc_gelu_poly",
    "enc_linear",
    "enc_ln_poly",
    "make_context",
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
    if name in {"enc_gelu_poly", "enc_linear", "enc_ln_poly"}:
        from . import ops

        return getattr(ops, name)
    raise AttributeError(f"module 'fhe_thesis.encryption' has no attribute {name!r}")
