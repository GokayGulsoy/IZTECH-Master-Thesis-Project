"""Encrypted compute primitives for Synthesizer-LPAN over CKKS.

Public API (focused submodules; nothing is imported eagerly so that
``import fhe_thesis.encryption`` is cheap when the GPU bindings are absent):

- ``backend``         -- abstract CKKSBackend interface and Ciphertext type.
- ``heongpu_backend`` -- HEonGPU (CUDA) implementation.
- ``colmajor``        -- column-major packing helpers + Galois key prep.
- ``multi``           -- multi-ciphertext arithmetic (per-bundle helpers).
- ``linear``          -- column-major linear projections (BSGS, streaming, multi).
- ``layernorm``       -- column-major LayerNorm (single-ct + multi-bundle).
- ``attention``       -- Synthesizer-LPAN attention (Tay 2020) -- first FHE port.
"""

__all__ = [
    "backend",
    "heongpu_backend",
    "colmajor",
    "multi",
    "linear",
    "layernorm",
    "attention",
]
