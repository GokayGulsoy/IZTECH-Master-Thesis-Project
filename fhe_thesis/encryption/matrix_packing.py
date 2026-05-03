"""Matrix-packed ciphertext layout for the LPAN-FHE protocol.

A :class:`MatrixPackedTensor` represents a 2-D activation tensor
``X ∈ R^{seq_len × hidden_dim}`` packed *several tokens per ciphertext*.

Layout
------
Let ``D = next_pow2(hidden_dim)`` be the per-token stride and
``B = num_slots // D`` the number of tokens that fit in one ciphertext.
The tensor is split into ``ceil(seq_len / B)`` ciphertexts; each
ciphertext holds the rows ``[b·B, b·B+1, …, b·B+B-1)`` of ``X`` placed
end-to-end::

    ct_b = [ X[bB,0..D),  X[bB+1,0..D),  …,  X[bB+B-1,0..D) ]

with the leading ``hidden_dim`` slots of every block holding the real
values and the trailing ``D - hidden_dim`` slots zero-padded so that
**slot-local element-wise ops** (``EvalPoly``, ``EvalAdd``,
``EvalMult``-with-broadcast-vector) work correctly across all *B*
sub-blocks at once.

Why this matters
----------------
Element-wise ops dominate ~30 % of LPAN-FHE wall time (GELU polyval,
softmax polyval, LN inv-sqrt, residual adds, plain-multiplies). With
``TokenPackedTensor`` each of those ops is paid per-token; with
matrix packing each call processes ``B`` tokens at once. For BERT-Base
at ``num_slots=4096`` and ``hidden_dim=768`` (``D=1024``), ``B=4`` —
already a 4× reduction on those ops with no extra rotations.

Note on rotations
-----------------
CKKS rotations are *cyclic over the entire ciphertext*. Slot-local ops
don't rotate so they are safe. Operations that *do* rotate (matmul,
inner-product, EvalSum) require a kernel that respects the block
boundaries; see :func:`enc_linear_matrix` in ``ops.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext


def next_pow2(n: int) -> int:
    """Smallest power of two ``≥ n``."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class MatrixPackedTensor:
    """Activation tensor packed B tokens per ciphertext.

    Attributes
    ----------
    cts : list of backend Ciphertext
        ``ceil(seq_len / B)`` ciphertexts.
    seq_len, hidden_dim : int
        Logical shape of the underlying tensor.
    block : int
        ``D = next_pow2(hidden_dim)`` — slot stride between sub-blocks.
    tokens_per_ct : int
        ``B = num_slots // block``.
    num_slots : int
        Backend slot count (replicated here for fast packing math).
    """

    cts: List[Ciphertext]
    seq_len: int
    hidden_dim: int
    block: int
    tokens_per_ct: int
    num_slots: int

    # ── factories ─────────────────────────────────────────────────────
    @classmethod
    def encrypt(
        cls,
        backend: CKKSBackend,
        x: np.ndarray,
        *,
        block: int = 0,
    ) -> "MatrixPackedTensor":
        """Encrypt a ``(seq_len, hidden_dim)`` tensor matrix-packed.

        Parameters
        ----------
        block : int, optional
            Per-token slot stride. ``0`` (default) → ``next_pow2(hidden_dim)``.
            Pass an explicit power of two to align with a downstream linear
            of larger ``out_dim`` (e.g. 4096 for BERT-Base FFN).
        """
        if x.ndim != 2:
            raise ValueError(f"expected 2-D tensor, got shape {x.shape}")
        seq_len, hidden_dim = x.shape
        n_slots = backend.capabilities.n_slots
        D = block if block > 0 else next_pow2(hidden_dim)
        if D < hidden_dim:
            raise ValueError(f"block={D} < hidden_dim={hidden_dim}")
        if D > n_slots:
            raise ValueError(
                f"block={D} exceeds backend slots={n_slots}; "
                f"use TokenPackedTensor instead"
            )
        B = n_slots // D
        if B < 1:
            raise ValueError(f"no tokens fit: block={D}, slots={n_slots}")

        cts: List[Ciphertext] = []
        n_groups = (seq_len + B - 1) // B
        for g in range(n_groups):
            buf = [0.0] * n_slots
            for k in range(B):
                tok = g * B + k
                if tok >= seq_len:
                    break
                row = x[tok]
                base = k * D
                # Real values; trailing (D-hidden_dim) slots stay zero.
                for j in range(hidden_dim):
                    buf[base + j] = float(row[j])
            cts.append(backend.encrypt(buf))
        return cls(
            cts=cts,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            block=D,
            tokens_per_ct=B,
            num_slots=n_slots,
        )

    @classmethod
    def from_token_packed(
        cls,
        backend: CKKSBackend,
        tpt,  # TokenPackedTensor — typed loosely to avoid import cycles
        *,
        block: int = 0,
    ) -> "MatrixPackedTensor":
        """Re-encrypt a TokenPackedTensor in matrix-packed layout.

        This decrypts every token, re-packs, and re-encrypts. Intended
        for *parity tests* and one-time ingestion at the protocol
        boundary; it is **not** meant for the hot path.
        """
        x = tpt.decrypt(backend)
        return cls.encrypt(backend, x, block=block)

    # ── decoding ──────────────────────────────────────────────────────
    def decrypt(self, backend: CKKSBackend) -> np.ndarray:
        """Decrypt back to a ``(seq_len, hidden_dim)`` numpy array."""
        out = np.zeros((self.seq_len, self.hidden_dim), dtype=np.float64)
        D = self.block
        B = self.tokens_per_ct
        for g, ct in enumerate(self.cts):
            slots = backend.decrypt(ct)
            for k in range(B):
                tok = g * B + k
                if tok >= self.seq_len:
                    break
                base = k * D
                out[tok, :] = slots[base : base + self.hidden_dim]
        return out

    def to_token_packed(self, backend: CKKSBackend):
        """Decrypt + re-encrypt to TokenPackedTensor (parity-test helper)."""
        # Local import avoids module-load cycle.
        from .packing import TokenPackedTensor

        return TokenPackedTensor.encrypt(backend, self.decrypt(backend))

    # ── shape helpers ─────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        return (self.seq_len, self.hidden_dim)

    def __len__(self) -> int:
        return len(self.cts)

    @classmethod
    def from_ciphertexts(
        cls,
        cts: Sequence[Ciphertext],
        *,
        seq_len: int,
        hidden_dim: int,
        block: int,
        tokens_per_ct: int,
        num_slots: int,
    ) -> "MatrixPackedTensor":
        """Wrap precomputed ciphertexts (e.g. output of an op)."""
        return cls(
            cts=list(cts),
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            block=block,
            tokens_per_ct=tokens_per_ct,
            num_slots=num_slots,
        )

    # ── convenience for slot-local ops in ops_matrix.py ───────────────
    def block_mask(self, hidden_dim: int) -> List[float]:
        """Return a length-``num_slots`` mask that is 1.0 in the first
        ``hidden_dim`` slots of every B-sub-block and 0.0 elsewhere.

        Used by element-wise ops to zero out the trailing pad of each
        sub-block when intermediate ops (e.g. polynomials with non-zero
        constant term) would otherwise leak nonzero values into the pad.
        """
        D = self.block
        B = self.tokens_per_ct
        mask = [0.0] * self.num_slots
        for k in range(B):
            base = k * D
            for j in range(hidden_dim):
                mask[base + j] = 1.0
        return mask

    def replicated_vector(self, vec: Sequence[float]) -> List[float]:
        """Replicate a length-``hidden_dim`` plaintext vector across the B sub-blocks.

        Used for per-feature plaintext operands like LN ``γ``/``β`` so the
        same logical multiplication / addition lands on every packed token.
        """
        if len(vec) != self.hidden_dim:
            raise ValueError(
                f"vec length {len(vec)} != hidden_dim {self.hidden_dim}"
            )
        D = self.block
        B = self.tokens_per_ct
        out = [0.0] * self.num_slots
        for k in range(B):
            base = k * D
            for j, v in enumerate(vec):
                out[base + j] = float(v)
        return out
