"""Encrypted compute primitives for Synthesizer-LPAN over CKKS.

Public API (all exported names live in submodules listed below):

- ``backend``           -- abstract CKKSBackend interface and Ciphertext type.
- ``heongpu_backend``   -- HEonGPU (CUDA) implementation.
- ``ops_attention_nexus`` -- packed-token operators (linear, layernorm,
  Synthesizer attention, multi-head bundles, BSGS variants).

Nothing is imported eagerly so that ``import fhe_thesis.encryption`` is cheap
even when the GPU bindings are unavailable.
"""

__all__ = [
    "backend",
    "heongpu_backend",
    "ops_attention_nexus",
]
