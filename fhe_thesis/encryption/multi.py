"""Multi-ciphertext arithmetic helpers (ct-bundles per layer)

Sliced from the original ``ops_attention_nexus.py`` during the production
re-modularization (synthesizer-lpan-production branch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext

# -------------------------------------------------------------------------
# Multi-ciphertext arithmetic helpers (ct-bundles per layer)
# -------------------------------------------------------------------------



def add_multi(backend, a_cts, b_cts):
    return [backend.add(a, b) for a, b in zip(a_cts, b_cts)]




def sub_multi(backend, a_cts, b_cts):
    return [backend.sub(a, b) for a, b in zip(a_cts, b_cts)]




def mul_multi(backend, a_cts, b_cts):
    return [backend.mul(a, b) for a, b in zip(a_cts, b_cts)]




def polyval_multi(backend, cts, coeffs):
    return [backend.polyval(c, list(coeffs)) for c in cts]




def per_col_sum_multi(
    backend: CKKSBackend,
    cts: Sequence[Ciphertext],
    *,
    L: int,
    hidden_per_ct: int,
    scale: float = 1.0,
) -> List[Ciphertext]:
    """Cross-ct sum: returns a list where every ct equals the broadcast of the
    full per-row sum (Σ over ALL hidden cols of all input cts) × scale.

    Each input ct is reduced internally with stride-L doubling, then partial
    broadcasts are summed. All output entries reference the same final ct
    (safe because downstream sub/mul return new cts).
    """
    n_slots = backend.capabilities.n_slots
    partials = [
        per_col_sum_then_broadcast(
            backend, c, L=L, hidden_dim=hidden_per_ct,
            num_slots=n_slots, scale=scale,
        )
        for c in cts
    ]
    total = partials[0]
    for p in partials[1:]:
        total = backend.add(total, p)
    return [total] * len(cts)


