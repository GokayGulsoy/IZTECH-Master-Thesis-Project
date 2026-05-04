"""Phase 7e: NEXUS-style matrix-aware diagonal attention.

Layout (column-major within each head):
    slot[j * L + i] = X[i, j]   for i ∈ [0, L), j ∈ [0, head_dim)

Where L is the (power-of-2) sequence length and head_dim is the per-head
feature width. One head fits in L * head_dim slots.

Why this layout matches NEXUS:
  • Q@K^T diagonal d:  S[i, (i+d) mod L] = Σ_j Q[i, j] · K[(i+d) mod L, j]
    - rotate K by d slots within each j-column (= rotate by d globally,
      because each column occupies L consecutive slots)
    - mul Q · rotated_K
    - log2(head_dim) doubling-sum across columns
    - → ONE ciphertext with S[i, (i+d) mod L] at slot i (within first
      column block); other slots zeroed by mask after sum.
  • No K-replication needed (slot 0..L is row 0..L of column 0;
    rotating by d gives row d..d+L mod L of column 0 — perfect cyclic).
  • No broadcast across head_dim (the sum already reduced it to col 0).
  • L iterations × (1 mul + log(head_dim) rots) ≈ 128 × 7 = ~900 ops
    for full BERT-base.

Compared to our previous diagonal kernel (Phase 7d):
    +  No 2× K replication (saves 2× slot budget)
    +  No L per-d slot masks (saves L mul_plain ops)
    +  No broadcast loop over head_dim (saves L × log(head_dim) rots)
    -  Different layout for Q/K/V — needs a repack from the linear's
       row-major output, OR a column-major-emitting linear projection
       (Phase 7e-3).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext


# ---------------------------------------------------------------------------
# Column-major packing helpers
# ---------------------------------------------------------------------------


def pack_colmajor(
    backend: CKKSBackend,
    X: np.ndarray,
    *,
    L: int,
    head_dim: int,
) -> Ciphertext:
    """Encrypt ``X[L, head_dim]`` in column-major layout.

    slot[j*L + i] = X[i, j]

    L must be a power of 2; pad input with zero rows if not.
    """
    if X.shape != (L, head_dim):
        raise ValueError(f"X.shape {X.shape} != ({L}, {head_dim})")
    if (L & (L - 1)) != 0:
        raise ValueError(f"L must be power of 2; got {L}")
    n_slots = backend.capabilities.n_slots
    if L * head_dim > n_slots:
        raise ValueError(f"L*head_dim={L*head_dim} > num_slots={n_slots}")
    buf = [0.0] * n_slots
    for j in range(head_dim):
        base = j * L
        for i in range(L):
            buf[base + i] = float(X[i, j])
    return backend.encrypt(buf)


def unpack_colmajor(
    backend: CKKSBackend,
    ct: Ciphertext,
    *,
    L: int,
    head_dim: int,
) -> np.ndarray:
    """Inverse of :func:`pack_colmajor`."""
    slots = backend.decrypt(ct)
    out = np.zeros((L, head_dim), dtype=np.float64)
    for j in range(head_dim):
        base = j * L
        for i in range(L):
            out[i, j] = slots[base + i]
    return out


# ---------------------------------------------------------------------------
# NEXUS-style diagonal Q@K^T
# ---------------------------------------------------------------------------


def qk_scores_nexus(
    backend: CKKSBackend,
    Q_ct: Ciphertext,
    K_ct: Ciphertext,
    *,
    L: int,
    head_dim: int,
    scale: float,
) -> Ciphertext:
    """NEXUS-style diagonal Q@K^T in column-major layout.

    Returns a single ct in **row-aligned diagonal layout**:
        slot[d * L + i] = scale · S[i, (i+d) mod L]   for d ∈ [0, L)

    (Conveniently, this is the same column-major shape as Q/K but where
    "j" is now the diagonal index d. So softmax / next stage can treat
    it as a column-major matrix of shape [L, L].)

    Algorithm:
      For d in 0..L-1:
        1. K_rot = rotate(K_ct, d)
           — Within each column [j*L, (j+1)*L), this cyclically rotates
             the L token rows. Result: K_rot[i + j*L] = K[(i+d) mod L, j].
             ✓ But: rotation by d slots wraps across column boundaries;
             specifically, the last d slots of column j-1 spill into the
             first d slots of column j. We must mask within each column.
        2. K_rot_masked = K_rot * within_col_mask(d)
           — Zero out the spill so each column independently shows
             K[(i+d)%L, j] for i ∈ [d, L), and 0 for i ∈ [0, d).
           Actually a cleaner approach: replicate K cyclically inside
           each column using a complementary mask + rotation.
        3. prod = Q_ct * K_rot
        4. Sum across head_dim columns (log2(head_dim) bare-doublings):
             for s in {L, 2L, 4L, ..., (head_dim/2)*L}:
                 prod += rotate(prod, s)
           After the loop, the first L slots hold Σ_j Q[i,j]*K[(i+d)%L,j]
           for i ∈ [0, L). Other column slots hold cyclic-shift garbage.
        5. mask to zero garbage outside first L slots, scale by 1/√d_h
        6. global rotate by -d*L → place at slots [d*L, (d+1)*L)
        7. accumulate into S

    Cost per d:
        1 rotation (K) + 1 mask mul_plain (cyclic correction)
        + 1 ct-mul (Q*K_rot)
        + log2(head_dim) bare-rotations (sum across cols)
        + 1 mask mul_plain + 1 global rotation (placement)
      = ~13 ops/d at head_dim=64
    Total: L * 13 ≈ 1700 ops for L=128 — vs ~16000 in our previous diag.
    ~10× faster on attention.
    """
    n_slots = backend.capabilities.n_slots
    if L * head_dim > n_slots:
        raise ValueError(f"L*head_dim={L*head_dim} > num_slots={n_slots}")

    S: Optional[Ciphertext] = None

    # Pre-build the "first L slots only" mask (no need to scale here; we
    # fold scale into the placement mask).
    first_L_mask = [0.0] * n_slots
    for i in range(L):
        first_L_mask[i] = 1.0

    for d in range(L):
        # Step 1+2: cyclic-shift K within each column by d.
        # Decomposition: K_shift_left = rotate(K, d); top d slots of col j
        # are now K[d..L, j], bottom L-d slots have K[0..d, j-1] spill.
        # K_shift_right = rotate(K, d - L); the spill is in the opposite
        # spots. Mask + add to get correct cyclic shift per column.
        if d == 0:
            K_rot = K_ct
        else:
            # within-col cyclic = mask(top L-d) of rotate(K, d)
            #                   + mask(bottom d)  of rotate(K, d - L)
            # Per-column. Since columns are L-aligned, we can build the
            # top-mask as: 1.0 in slots [j*L, j*L + L - d), 0 elsewhere.
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(head_dim):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            K_left  = backend.rotate(K_ct, d)        # shift up
            K_right = backend.rotate(K_ct, d - L)    # shift down by L-d
            K_rot = backend.add(
                backend.mul_plain(K_left, top_mask),
                backend.mul_plain(K_right, bot_mask),
            )

        # Step 3: Q * K_rot
        prod = backend.mul(Q_ct, K_rot)

        # Step 4: sum across head_dim columns via L-stride bare-doublings.
        sum_ct = prod
        step = L
        while step < L * head_dim:
            sum_ct = backend.add(sum_ct, backend.rotate(sum_ct, step))
            step <<= 1
        # After the loop, first L slots have Σ_j Q[i,j]*K[(i+d)%L,j].
        # Other slots have cyclic-shift garbage.

        # Step 5+6: scale + place at slots [d*L, (d+1)*L).
        # Build a mask: 1.0 in slots [0, L), 0 elsewhere, scaled by `scale`.
        # Then global rotate by -d*L (right-shift by d*L) to place.
        place_mask = [0.0] * n_slots
        for i in range(L):
            place_mask[i] = float(scale)
        masked = backend.mul_plain(sum_ct, place_mask)
        if d > 0:
            masked = backend.rotate(masked, -d * L)

        S = masked if S is None else backend.add(S, masked)

    if S is None:
        S = backend.mul_plain(Q_ct, [0.0] * n_slots)
    return S
