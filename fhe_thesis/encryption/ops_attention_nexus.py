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


# ---------------------------------------------------------------------------
# NEXUS-style diagonal A@V
# ---------------------------------------------------------------------------


def attn_apply_nexus(
    backend: CKKSBackend,
    A_ct: Ciphertext,
    V_ct: Ciphertext,
    *,
    L: int,
    head_dim: int,
) -> Ciphertext:
    """NEXUS-style diagonal A @ V in column-major layout.

    Inputs:
      A_ct: in diagonal-row layout (output of qk_scores_nexus).
            slot[d*L + i] = A[i, (i+d) mod L]
      V_ct: column-major.  slot[j*L + i] = V[i, j]

    Output:
      Out_ct: column-major.  slot[j*L + i] = Out[i, j]
              where  Out[i, j] = Σ_k A[i, k] * V[k, j]
                              = Σ_d A[i, (i+d) mod L] * V[(i+d) mod L, j]

    Algorithm per d ∈ [0, L):
      1. Extract row d of A: rotate(A, d*L) + mask first L slots
         → slot i (for i ∈ [0, L)) holds A[i, (i+d) mod L].
      2. Broadcast across head_dim columns via L-stride bare-doublings.
      3. V_shift_d: within-column cyclic shift of V by d (same trick
         as in qk_scores_nexus).
      4. prod = a_row_bcast * V_shift_d
      5. accumulate.

    Cost per d: ~14 ops. Total: L * 14 ≈ 1800 ops/head at L=128.
    Same complexity as qk_scores_nexus.
    """
    n_slots = backend.capabilities.n_slots
    if L * head_dim > n_slots:
        raise ValueError(f"L*head_dim={L*head_dim} > num_slots={n_slots}")

    Out: Optional[Ciphertext] = None

    # Pre-build the "first L slots" mask (used to isolate row d after
    # the global rotate).
    first_L_mask = [0.0] * n_slots
    for i in range(L):
        first_L_mask[i] = 1.0

    for d in range(L):
        # Step 1: bring row d of A to slots [0, L).
        if d == 0:
            a_row = A_ct
        else:
            a_row = backend.rotate(A_ct, d * L)
        a_row = backend.mul_plain(a_row, first_L_mask)

        # Step 2: broadcast row d across all head_dim columns.
        bcast = a_row
        step = L
        while step < L * head_dim:
            bcast = backend.add(bcast, backend.rotate(bcast, -step))
            step <<= 1
        # bcast now has the row-d values replicated in every L-stride
        # column block: slot[j*L + i] = A[i, (i+d) mod L] for all j.

        # Step 3: within-column cyclic shift of V by d (same as QK^T).
        if d == 0:
            V_shift = V_ct
        else:
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(head_dim):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            V_left  = backend.rotate(V_ct, d)
            V_right = backend.rotate(V_ct, d - L)
            V_shift = backend.add(
                backend.mul_plain(V_left, top_mask),
                backend.mul_plain(V_right, bot_mask),
            )

        # Step 4+5: product and accumulate.
        prod = backend.mul(bcast, V_shift)
        Out = prod if Out is None else backend.add(Out, prod)

    if Out is None:
        Out = backend.mul_plain(V_ct, [0.0] * n_slots)
    return Out


# ---------------------------------------------------------------------------
# Column-major linear projection (Halevi-Shoup with stride L)
# ---------------------------------------------------------------------------


def linear_colmajor(
    backend: CKKSBackend,
    x_ct: Ciphertext,
    W: np.ndarray,
    *,
    L: int,
    in_dim: int,
    out_dim: int,
    bias: Optional[np.ndarray] = None,
) -> Ciphertext:
    """Halevi-Shoup linear projection in column-major slot layout.

    Input  ct slot[j*L + i] = X[i, j]  for j ∈ [0, in_dim)
    Output ct slot[j*L + i] = Y[i, j] = Σ_k W[j, k] · X[i, k]
            for j ∈ [0, out_dim) (slots beyond out_dim*L are zero).

    Algorithm:
      For d ∈ [0, in_dim):
        x_rot   = rotate(x_ct, d * L)   # cyclically shifts whole columns
        diag_d  = plaintext: slot[j*L + i] = W[j, (j+d) mod in_dim]
                  for j ∈ [0, out_dim), i ∈ [0, L); 0 elsewhere
        y      += mul_plain(x_rot, diag_d)
      add bias if given (bias[j] replicated across slots [j*L, j*L+L))

    Cost: in_dim mul_plain + in_dim rotations + in_dim adds.
    Depth: +1 (single mul_plain rescale).

    Constraint: in_dim * L ≤ num_slots AND out_dim * L ≤ num_slots.
    """
    n_slots = backend.capabilities.n_slots
    if W.shape != (out_dim, in_dim):
        raise ValueError(f"W.shape {W.shape} != ({out_dim}, {in_dim})")
    if max(in_dim, out_dim) * L > n_slots:
        raise ValueError(
            f"max({in_dim},{out_dim}) * L = {max(in_dim, out_dim) * L} "
            f"> num_slots {n_slots}"
        )

    # For the global-rotate-by-d*L cyclic-shift trick to be correct, we
    # need the active data region (in_dim · L) to evenly divide num_slots
    # AND to be replicated to fill num_slots. Otherwise a rotation by d*L
    # for d near in_dim wraps into trailing zero slots instead of cycling
    # back to column 0.
    #
    # Solution: round in_dim up to in_dim_padded (next power of 2 dividing
    # num_slots/L) and zero-pad. Then replicate to fill num_slots so the
    # global rotation is naturally cyclic over in_dim_padded.
    max_cols = n_slots // L
    if max_cols * L != n_slots or (max_cols & (max_cols - 1)) != 0:
        raise ValueError(
            f"L={L} doesn't cleanly divide num_slots={n_slots} into a "
            f"power-of-2 column count"
        )
    in_dim_padded = 1
    while in_dim_padded < in_dim:
        in_dim_padded <<= 1
    if in_dim_padded > max_cols:
        raise ValueError(
            f"in_dim_padded={in_dim_padded} > max_cols={max_cols}"
        )

    # Replicate x_ct to fill num_slots: x_ct holds in_dim valid columns
    # in slots [0, in_dim*L); we want the same data periodically repeated
    # so that slot (j+d)*L + i for any j,d hits a valid column value.
    # Doubling-add with rotation: add(x, rotate(x, -in_dim_padded*L * 2^k))
    # log2(num_slots / (in_dim_padded*L)) times.
    x_replicated = x_ct
    cur = in_dim_padded * L
    while cur < n_slots:
        x_replicated = backend.add(
            x_replicated, backend.rotate(x_replicated, -cur)
        )
        cur <<= 1

    n = in_dim_padded  # number of diagonals
    Y: Optional[Ciphertext] = None
    for d in range(n):
        if d == 0:
            x_rot = x_replicated
        else:
            x_rot = backend.rotate(x_replicated, d * L)
        diag = [0.0] * n_slots
        any_nz = False
        for j in range(out_dim):
            col_src = (j + d) % n
            if col_src >= in_dim:
                continue  # padded column, weight is implicitly zero
            w = float(W[j, col_src])
            if w == 0.0:
                continue
            any_nz = True
            base = j * L
            for i in range(L):
                diag[base + i] = w
        if not any_nz:
            continue
        term = backend.mul_plain(x_rot, diag)
        Y = term if Y is None else backend.add(Y, term)

    if Y is None:
        Y = backend.mul_plain(x_ct, [0.0] * n_slots)

    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"bias.shape {bias.shape} != ({out_dim},)")
        bvec = [0.0] * n_slots
        for j in range(out_dim):
            base = j * L
            for i in range(L):
                bvec[base + i] = float(bias[j])
        Y = backend.add_plain(Y, bvec)

    return Y
