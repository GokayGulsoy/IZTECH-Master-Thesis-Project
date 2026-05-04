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
    num_heads_per_ct: int = 1,
) -> Ciphertext:
    """NEXUS-style diagonal Q@K^T in column-major layout.

    With ``num_heads_per_ct > 1``: Q and K each pack multiple heads side by
    side at slot ranges ``[h·head_dim·L, (h+1)·head_dim·L)`` for h ∈
    [0, num_heads_per_ct). The output similarly packs S for each head
    (in row-aligned diagonal layout) at the same slot ranges.

    Returns a single ct in **row-aligned diagonal layout**:
        slot[h*H + d * L + i] = scale · S_h[i, (i+d) mod L]
        where H = head_dim·L, for d ∈ [0, L), h ∈ [0, num_heads_per_ct).
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    total = num_heads_per_ct * H
    if total > n_slots:
        raise ValueError(
            f"num_heads_per_ct·L·head_dim={total} > num_slots={n_slots}"
        )

    S: Optional[Ciphertext] = None
    n_cols = num_heads_per_ct * head_dim  # total columns across all heads

    for d in range(L):
        # Step 1+2: cyclic-shift K within each column by d (across ALL columns).
        if d == 0:
            K_rot = K_ct
        else:
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(n_cols):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            K_left  = backend.rotate(K_ct, d)
            K_right = backend.rotate(K_ct, d - L)
            K_rot = backend.add(
                backend.mul_plain(K_left, top_mask),
                backend.mul_plain(K_right, bot_mask),
            )

        # Step 3: Q * K_rot
        prod = backend.mul(Q_ct, K_rot)

        # Step 4: per-head sum across head_dim columns (stop at H = head_dim·L
        # so each head's reduction stays inside its own region).
        sum_ct = prod
        step = L
        while step < H:
            sum_ct = backend.add(sum_ct, backend.rotate(sum_ct, step))
            step <<= 1

        # Step 5+6: place mask scaled by `scale`, replicated per head.
        # 1.0 in slots [h*H, h*H + L) for h ∈ [0, num_heads_per_ct).
        place_mask = [0.0] * n_slots
        for h in range(num_heads_per_ct):
            base = h * H
            for i in range(L):
                place_mask[base + i] = float(scale)
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
    num_heads_per_ct: int = 1,
) -> Ciphertext:
    """NEXUS-style diagonal A @ V in column-major layout.

    Multi-head: A and V each pack ``num_heads_per_ct`` heads in adjacent
    slot regions of width ``head_dim·L``; output is similarly packed.
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    total = num_heads_per_ct * H
    if total > n_slots:
        raise ValueError(
            f"num_heads_per_ct·L·head_dim={total} > num_slots={n_slots}"
        )

    Out: Optional[Ciphertext] = None
    n_cols = num_heads_per_ct * head_dim

    # Per-head "first L" mask: 1.0 in slots [h*H, h*H + L) for each head.
    first_L_mask = [0.0] * n_slots
    for h in range(num_heads_per_ct):
        base = h * H
        for i in range(L):
            first_L_mask[base + i] = 1.0

    for d in range(L):
        # Step 1: bring row d of each head's A to slots [h*H, h*H + L).
        if d == 0:
            a_row = A_ct
        else:
            a_row = backend.rotate(A_ct, d * L)
        a_row = backend.mul_plain(a_row, first_L_mask)

        # Step 2: broadcast row d across head_dim cols PER HEAD (stop at H).
        bcast = a_row
        step = L
        while step < H:
            bcast = backend.add(bcast, backend.rotate(bcast, -step))
            step <<= 1

        # Step 3: within-column cyclic shift of V by d (across all cols).
        if d == 0:
            V_shift = V_ct
        else:
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(n_cols):
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
# Rotation-key pre-registration (avoids bit-decomp rotations)
# ---------------------------------------------------------------------------


def prepare_colmajor_keys(
    backend: CKKSBackend,
    *,
    L: int,
    max_dim: int,
) -> int:
    """Pre-register every rotation step needed by the col-major kernels.

    With BSGS in ``linear_colmajor`` we only need ``bs + gs ≈ 2·sqrt(max_dim)``
    distinct shifts (positive and negative) instead of ``max_dim`` of them.
    Also covers ``per_col_sum_then_broadcast`` which uses ±L, ±2L, ±4L, ...

    For BERT-base hidden_padded=1024 → bs=gs=32 → 62 keys × ~32 MB ≈ 2 GB
    (vs 1023 keys × 32 MB ≈ 33 GB in the naive variant — which OOMs on H100).

    Returns the number of *new* shifts registered.
    """
    if not hasattr(backend, "register_rotation_keys"):
        return 0
    n = 1
    while n < max_dim:
        n <<= 1
    log2_n = n.bit_length() - 1
    bs = 1 << (log2_n // 2)
    gs = n // bs
    shifts = set()
    # BSGS baby/giant for linear_colmajor
    for i in range(1, bs):
        shifts.add(i * L)
        shifts.add(-i * L)
    for j in range(1, gs):
        shifts.add(j * bs * L)
        shifts.add(-j * bs * L)
    # per_col_sum_then_broadcast uses ±L, ±2L, ±4L, ..., ±n/2 · L
    s = L
    while s < n * L:
        shifts.add(s)
        shifts.add(-s)
        s <<= 1
    return backend.register_rotation_keys(sorted(shifts))


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
    bsgs: bool = True,
) -> Ciphertext:
    """Halevi-Shoup linear projection in column-major slot layout.

    Input  ct slot[j*L + i] = X[i, j]  for j ∈ [0, in_dim)
    Output ct slot[j*L + i] = Y[i, j] = Σ_k W[j, k] · X[i, k]
            for j ∈ [0, out_dim) (slots beyond out_dim*L are zero).

    With ``bsgs=True`` (default): uses baby-step / giant-step factorisation
    n = bs · gs (with bs = gs = sqrt(n)) so only ``bs + gs − 2`` rotations
    and ``bs + gs`` rotation keys are required (vs ``n`` of each in the
    naive HS formulation). For BERT-base hidden=768 → n_padded=1024 →
    bs = gs = 32, so 62 rotations / 62 keys instead of 1023 / 1023.

    BSGS identity (cyclic):
        Σ_{d=0}^{n-1} rot(x, d·L) ⊙ diag_d
      = Σ_j rot( Σ_i rot(x, i·L) ⊙ rot(diag_{i+j·bs}, -j·bs·L), j·bs·L )

    Cost: n mul_plain + (bs+gs−2) rotations + n adds.
    Depth: +1 (single mul_plain rescale).

    Constraint: in_dim_padded · L ≤ num_slots, max_cols = num_slots/L
    must be a power of 2.
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

    def _build_diag(d: int) -> List[float]:
        """Plaintext for diagonal d: slot[j*L + i] = W[j, (j+d) mod in_dim]
        for j ∈ [0, out_dim), 0 elsewhere (and 0 if (j+d) is in padded region)."""
        v = [0.0] * n_slots
        for j in range(out_dim):
            col_src = (j + d) % n
            if col_src >= in_dim:
                continue
            w = float(W[j, col_src])
            if w == 0.0:
                continue
            base = j * L
            for i in range(L):
                v[base + i] = w
        return v

    def _rot_left_plain(v: List[float], k: int) -> List[float]:
        """Return v'[m] = v[(m + k) mod n_slots] (left rotation by k slots)."""
        k %= n_slots
        if k == 0:
            return v
        return v[k:] + v[:k]

    Y: Optional[Ciphertext] = None

    if not bsgs or n <= 4:
        # Naive HS path (kept for tiny test cases / sanity checks).
        for d in range(n):
            x_rot = x_replicated if d == 0 else backend.rotate(x_replicated, d * L)
            diag = _build_diag(d)
            if not any(diag):
                continue
            term = backend.mul_plain(x_rot, diag)
            Y = term if Y is None else backend.add(Y, term)
    else:
        # BSGS: pick bs = gs = sqrt(n) (n is a power of 2 ⇒ bs = 2^(log2(n)//2)).
        log2_n = n.bit_length() - 1
        bs = 1 << (log2_n // 2)
        gs = n // bs
        # Pre-compute baby rotations of x: rot(x, i*L) for i ∈ [0, bs).
        x_baby: List[Ciphertext] = [x_replicated]
        for i in range(1, bs):
            x_baby.append(backend.rotate(x_replicated, i * L))
        # Outer giant loop.
        for j in range(gs):
            inner: Optional[Ciphertext] = None
            shift = -j * bs * L  # right-shift the diag plaintext by j*bs*L
            for i in range(bs):
                d = i + j * bs
                diag = _build_diag(d)
                if not any(diag):
                    continue
                # rot(diag, -j*bs*L) at the plaintext level = left-rotate by shift.
                # Left rotation by negative k = right rotation by |k|.
                diag_shifted = _rot_left_plain(diag, shift)
                term = backend.mul_plain(x_baby[i], diag_shifted)
                inner = term if inner is None else backend.add(inner, term)
            if inner is None:
                continue
            if j != 0:
                inner = backend.rotate(inner, j * bs * L)
            Y = inner if Y is None else backend.add(Y, inner)

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


# ---------------------------------------------------------------------------
# Column-major LayerNorm (stride-L per-token reduction)
# ---------------------------------------------------------------------------


def per_col_sum_then_broadcast(
    backend: CKKSBackend,
    ct: Ciphertext,
    *,
    L: int,
    hidden_dim: int,
    num_slots: int,
    scale: float = 1.0,
) -> Ciphertext:
    """Sum across the ``hidden_dim`` columns of a column-major ct, then
    broadcast the per-row sum back to the same column-major layout.

    Input  : slot[j*L + i] = X[i, j]  for j ∈ [0, hidden_dim)
    Output : slot[j*L + i] = scale · Σ_j X[i, j]  for the same range.

    ``hidden_dim`` is padded internally to the next power of 2; pad columns
    must already be zero (true after ``pack_colmajor`` or after a previous
    masked op). The caller's ``scale`` should reflect the *real* hidden_dim
    (e.g. ``1/768`` for BERT-base, not ``1/1024``).

    Algorithm (analogue of :func:`per_block_sum_then_broadcast` but
    with stride ``L`` instead of stride 1):

      1. Bare doubling sum over stride-L offsets ``L, 2L, 4L, ...``.
         After ``log2(hidden_padded)`` iterations the first L slots
         hold the per-row sum; other slots are cyclic-shift garbage.
      2. Single mul_plain by the "first-L-slots" mask scaled by ``scale``.
      3. Bare doubling broadcast with right-rotations by
         ``L, 2L, 4L, ...`` for ``log2(hidden_padded)`` iterations.

    Total depth: 1 level. Constraint: ``hidden_padded · L ≤ num_slots``.
    """
    h_pad = 1
    while h_pad < hidden_dim:
        h_pad <<= 1
    if h_pad * L > num_slots:
        raise ValueError(
            f"hidden_padded*L={h_pad * L} exceeds num_slots={num_slots}"
        )

    # 1. Bare doubling sum.
    out = ct
    step = L
    while step < h_pad * L:
        out = backend.add(out, backend.rotate(out, step))
        step <<= 1

    # 2. First-L mask scaled by ``scale``.
    mask = [0.0] * num_slots
    for i in range(L):
        mask[i] = float(scale)
    out = backend.mul_plain(out, mask)

    # 3. Bare doubling broadcast with right-rotations.
    step = L
    while step < h_pad * L:
        out = backend.add(out, backend.rotate(out, -step))
        step <<= 1
    return out


def layernorm_colmajor(
    backend: CKKSBackend,
    x_ct: Ciphertext,
    *,
    L: int,
    hidden_dim: int,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: tuple,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> Ciphertext:
    """LPAN polynomial LayerNorm in the column-major slot layout.

    For each row i ∈ [0, L):
        μ_i  = (1/h) Σ_j X[i, j]
        σ²_i = (1/h) Σ_j (X[i, j] − μ_i)²
        Y[i, j] = γ_j · (X[i, j] − μ_i) · poly_invsqrt(σ²_i + eps) + β_j

    The eps is folded into ``invsqrt_interval`` by the caller (or is
    implicitly zero for the trained polynomial). γ/β are per-feature
    (per-column) constants.

    Depth: 1 (mean) + 1 (square) + 1 (var mul_plain) + invsqrt_poly_depth
    + 2 (gamma·(x-μ)·invσ) ≈ matches the row-major version.
    """
    h = hidden_dim
    inv_h = 1.0 / h
    n_slots = backend.capabilities.n_slots

    # Internal padding to next power of 2 (per_col_sum_then_broadcast handles
    # this transparently). Pad slots are assumed zero on input.
    h_pad = 1
    while h_pad < h:
        h_pad <<= 1

    a, b = invsqrt_interval
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    # Affine absorb: poly evaluated on (s·x + sh) instead of x.
    n = len(invsqrt_power_coeffs)
    absorbed = [0.0] * n
    # (s·x + sh)^k  expanded via binomial.
    from math import comb
    for k, c in enumerate(invsqrt_power_coeffs):
        for r in range(k + 1):
            absorbed[r] += float(c) * (sscale ** r) * (sshift ** (k - r)) * comb(k, r)

    # γ / β replicated col-major: slot[j*L + i] = γ[j] / β[j].
    gamma_vec = [0.0] * n_slots
    beta_vec = [0.0] * n_slots
    for j in range(h):
        base = j * L
        for i in range(L):
            gamma_vec[base + i] = float(gamma[j])
            beta_vec[base + i] = float(beta[j])

    # μ broadcast.
    mean_bc = per_col_sum_then_broadcast(
        backend, x_ct, L=L, hidden_dim=h, num_slots=n_slots, scale=inv_h,
    )
    centred = backend.sub(x_ct, mean_bc)
    # When h is not a power of 2, mean_bc spans the padded region too, so
    # centred has -μ in pad slots. Mask them off before squaring so the
    # variance sum is correct. (No-op when h == h_pad.)
    if h_pad != h:
        active_mask = [0.0] * n_slots
        for j in range(h):
            base = j * L
            for i in range(L):
                active_mask[base + i] = 1.0
        centred = backend.mul_plain(centred, active_mask)
    sq = backend.mul(centred, centred)
    var_bc = per_col_sum_then_broadcast(
        backend, sq, L=L, hidden_dim=h, num_slots=n_slots, scale=inv_h,
    )
    inv_sigma = backend.polyval(var_bc, absorbed)
    scaled = backend.mul(centred, inv_sigma)
    scaled = backend.mul_plain(scaled, gamma_vec)
    out = backend.add_plain(scaled, beta_vec)
    return out
