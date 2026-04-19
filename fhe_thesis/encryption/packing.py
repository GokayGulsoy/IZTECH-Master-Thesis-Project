"""Token-packed ciphertext layout for the LPAN-FHE protocol.

A `TokenPackedTensor` represents a 2-D activation tensor
`X ∈ R^{seq_len × hidden_dim}` as a list of `seq_len` ciphertexts,
each holding one token's hidden_dim values in its first hidden_dim
slots.

This is the canonical layout assumed by every operation in `ops.py`.
See `docs/ckks_protocol.md` §4 for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext


@dataclass
class TokenPackedTensor:
    """List of ciphertexts, one per token row."""

    cts: List[Ciphertext]
    seq_len: int
    hidden_dim: int

    # ── factory ───────────────────────────────────────────────────────
    @classmethod
    def encrypt(cls, backend: CKKSBackend, x: np.ndarray) -> "TokenPackedTensor":
        """Encrypt a (seq_len, hidden_dim) numpy array."""
        if x.ndim != 2:
            raise ValueError(f"expected 2-D tensor, got shape {x.shape}")
        seq_len, hidden_dim = x.shape
        if hidden_dim > backend.capabilities.n_slots:
            raise ValueError(
                f"hidden_dim={hidden_dim} exceeds backend slot count "
                f"{backend.capabilities.n_slots}"
            )
        cts = [backend.encrypt(x[i].tolist()) for i in range(seq_len)]
        return cls(cts=cts, seq_len=seq_len, hidden_dim=hidden_dim)

    # ── decoding ──────────────────────────────────────────────────────
    def decrypt(self, backend: CKKSBackend) -> np.ndarray:
        rows = [backend.decrypt(ct)[: self.hidden_dim] for ct in self.cts]
        return np.array(rows, dtype=np.float64)

    # ── shape helpers ─────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        return (self.seq_len, self.hidden_dim)

    def __len__(self) -> int:
        return self.seq_len

    def row(self, i: int) -> Ciphertext:
        return self.cts[i]

    # ── convenience constructors ──────────────────────────────────────
    @classmethod
    def from_ciphertexts(
        cls,
        cts: Sequence[Ciphertext],
        hidden_dim: int,
    ) -> "TokenPackedTensor":
        cts = list(cts)
        return cls(cts=cts, seq_len=len(cts), hidden_dim=hidden_dim)
