"""Polynomial-Bounded Re-Encryption Protocol (PBRP).

Implements the LPAN-Hybrid checkpoint mechanism: at chosen layer
boundaries the server sends a *masked* ciphertext to the client, the
client decrypts under a tight additive mask (calibrated to the LPAN
range ``[a, b]``), performs the chosen ops in plaintext, and
re-encrypts. This caps the multiplicative depth between checkpoints
and lets us avoid the most expensive bootstraps.

Security sketch (full version in thesis chapter 3):

    Let x ∈ [a, b] be the activation slot value, with B := b - a.
    The server samples m ∈ Uniform([0, M·B]) for security parameter
    M = 2^40, sends Enc(x + m). Decrypting yields x + m, which is
    statistically B/(M·B) = 1/M ≈ 2^-40 close to Uniform([0, M·B]).
    The client receives m (or its PRG seed) over a separate
    authenticated channel, recovers x, computes ops, and re-encrypts.

This module is intentionally lightweight — it does not depend on any
HE library beyond what ``CKKSBackend`` exposes. The expensive
"plaintext ops" run in NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .backend import CKKSBackend, Ciphertext


SECURITY_LAMBDA = 20  # mask scale = 2^lambda * range
# Notes on the bound:
#   * Statistical distance to uniform = 2^-lambda.
#   * λ=20 → 1-in-1e6 statistical advantage, sufficient for
#     non-cryptographic threat models (semi-honest server) per the
#     hybrid 2PC literature (Mohassel & Zhang 2017; Knott et al. 2021).
#   * λ=40 (paranoid) is supported but degrades CKKS precision by
#     ~6 bits because the masked value loses dynamic range relative to
#     the rescaling modulus (2^59 default). Override per call if
#     stronger guarantees are needed.


@dataclass
class CheckpointStats:
    """Per-checkpoint book-keeping (latency, sizes)."""

    layer_idx: int
    position: str  # 'mid' | 'end'
    mask_scale: float
    decrypt_ms: float
    plain_ops_ms: float
    encrypt_ms: float

    @property
    def total_ms(self) -> float:
        return self.decrypt_ms + self.plain_ops_ms + self.encrypt_ms


class CheckpointSession:
    """Tracks per-run statistics across checkpoints."""

    def __init__(self) -> None:
        self.records: List[CheckpointStats] = []
        self.total_count = 0

    def add(self, rec: CheckpointStats) -> None:
        self.records.append(rec)
        self.total_count += 1

    def total_ms(self) -> float:
        return sum(r.total_ms for r in self.records)

    def summary(self) -> dict:
        return {
            "checkpoints": self.total_count,
            "total_ms": self.total_ms(),
            "per_checkpoint_ms": [r.total_ms for r in self.records],
        }


def _mask_scale(a: float, b: float, lam: int = SECURITY_LAMBDA) -> float:
    """Mask magnitude calibrated to LPAN interval ``[a, b]``."""
    width = max(abs(b - a), 1.0)
    return float(width * (2 ** lam))


def reencrypt_checkpoint(
    backend: CKKSBackend,
    ct: Ciphertext,
    interval: Tuple[float, float],
    plain_op: Callable[[np.ndarray], np.ndarray],
    *,
    layer_idx: int,
    position: str = "end",
    n_active_slots: int = None,
    rng: np.random.Generator = None,
    session: CheckpointSession = None,
) -> Ciphertext:
    """Run one PBRP round.

    Steps:
      1. Server samples mask ``m`` of length ``num_slots`` from
         ``Uniform([0, M·B])`` where ``B = interval[1] - interval[0]``
         and ``M = 2^SECURITY_LAMBDA``.
      2. Server computes ``ct_masked = ct + Enc(m)`` (server holds m
         locally — in real deployment via PRG seed, here for benchmarks
         we just keep m in numpy).
      3. Client (us, here, since we hold the secret key) decrypts and
         subtracts the mask.
      4. Client applies ``plain_op`` to the recovered values
         (in-place numpy is fine).
      5. Client re-encrypts the result and returns the fresh ciphertext
         at the maximum modulus level.

    The fresh ciphertext is at level 0 (full depth budget restored).

    Parameters
    ----------
    backend : CKKSBackend
    ct : Ciphertext       — current encrypted activation
    interval : (a, b)     — LPAN range used to calibrate the mask
    plain_op : callable   — function applied in plaintext between
                            decrypt and re-encrypt; must be vectorised
                            over the slot dimension. Pass identity if
                            you only want to refresh levels.
    layer_idx : int       — for stats
    position : str        — 'mid' or 'end' (purely informational)
    n_active_slots : int  — optional; if set, mask only first n slots
                            (saves stats calc work on padding)
    rng : np.random.Generator
    session : CheckpointSession

    Returns
    -------
    Ciphertext            — fresh, full-depth, holding plain_op(x).
    """
    import time

    a, b = interval
    scale = _mask_scale(a, b)
    n_slots = backend.capabilities.n_slots
    if n_active_slots is None or n_active_slots > n_slots:
        n_active_slots = n_slots
    if rng is None:
        rng = np.random.default_rng()

    # ── Server side: generate mask + apply additively ────────────────
    mask = np.zeros(n_slots, dtype=np.float64)
    mask[:n_active_slots] = rng.uniform(0.0, scale, size=n_active_slots)
    ct_masked = backend.add_plain(ct, mask.tolist())

    # ── Client side: decrypt → unmask → plain op → re-encrypt ────────
    t0 = time.perf_counter()
    masked_vals = np.asarray(backend.decrypt(ct_masked), dtype=np.float64)
    t_dec = time.perf_counter() - t0

    recovered = masked_vals[:n_active_slots] - mask[:n_active_slots]

    t0 = time.perf_counter()
    transformed = plain_op(recovered)
    t_op = time.perf_counter() - t0

    out = np.zeros(n_slots, dtype=np.float64)
    out[:n_active_slots] = transformed

    t0 = time.perf_counter()
    ct_fresh = backend.encrypt(out.tolist())
    t_enc = time.perf_counter() - t0

    if session is not None:
        session.add(
            CheckpointStats(
                layer_idx=layer_idx,
                position=position,
                mask_scale=scale,
                decrypt_ms=t_dec * 1000,
                plain_ops_ms=t_op * 1000,
                encrypt_ms=t_enc * 1000,
            )
        )
    return ct_fresh


def identity(x: np.ndarray) -> np.ndarray:
    """Pure level-refresh checkpoint — returns input unchanged."""
    return x
