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
    cache: Optional[dict] = None,
) -> Ciphertext:
    """NEXUS-style diagonal Q@K^T in column-major layout.

    With ``num_heads_per_ct > 1``: Q and K each pack multiple heads side by
    side at slot ranges ``[h·head_dim·L, (h+1)·head_dim·L)`` for h ∈
    [0, num_heads_per_ct). The output similarly packs S for each head
    (in row-aligned diagonal layout) at the same slot ranges.

    ``cache`` (optional dict): if provided, top/bot/place masks are
    encoded once and stored in this dict; subsequent calls with the same
    geometry reuse the encoded plaintexts. The cache is keyed by
    (L, head_dim, num_heads_per_ct, scale) for safety.

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

    n_cols = num_heads_per_ct * head_dim
    cache_key = ("qk", L, head_dim, num_heads_per_ct, scale)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        top_pts = cached["top_pts"]
        bot_pts = cached["bot_pts"]
        place_pt = cached["place_pt"]
    else:
        top_pts: List = [None]  # index by d; d=0 unused
        bot_pts: List = [None]
        for d in range(1, L):
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(n_cols):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            top_pts.append(backend._encode(top_mask))
            bot_pts.append(backend._encode(bot_mask))
        # place mask: scale at slots [h*H, h*H+L) for each head.
        pmask = [0.0] * n_slots
        for h in range(num_heads_per_ct):
            base = h * H
            for i in range(L):
                pmask[base + i] = float(scale)
        place_pt = backend._encode(pmask)
        if cache is not None:
            cache[cache_key] = {
                "top_pts": top_pts,
                "bot_pts": bot_pts,
                "place_pt": place_pt,
            }

    S: Optional[Ciphertext] = None

    for d in range(L):
        # Step 1+2: cyclic-shift K within each column by d (across ALL columns).
        if d == 0:
            K_rot = K_ct
        else:
            K_left  = backend.rotate(K_ct, d)
            K_right = backend.rotate(K_ct, d - L)
            K_rot = backend.add(
                backend._mul_plain_pt(K_left, top_pts[d]),
                backend._mul_plain_pt(K_right, bot_pts[d]),
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
        masked = backend._mul_plain_pt(sum_ct, place_pt)
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
    cache: Optional[dict] = None,
) -> Ciphertext:
    """NEXUS-style diagonal A @ V in column-major layout.

    Multi-head: A and V each pack ``num_heads_per_ct`` heads in adjacent
    slot regions of width ``head_dim·L``; output is similarly packed.

    ``cache``: same convention as :func:`qk_scores_nexus`.
    """
    n_slots = backend.capabilities.n_slots
    H = head_dim * L
    total = num_heads_per_ct * H
    if total > n_slots:
        raise ValueError(
            f"num_heads_per_ct·L·head_dim={total} > num_slots={n_slots}"
        )

    n_cols = num_heads_per_ct * head_dim
    cache_key = ("av", L, head_dim, num_heads_per_ct)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        first_L_pt = cached["first_L_pt"]
        top_pts = cached["top_pts"]
        bot_pts = cached["bot_pts"]
    else:
        first_L_mask = [0.0] * n_slots
        for h in range(num_heads_per_ct):
            base = h * H
            for i in range(L):
                first_L_mask[base + i] = 1.0
        first_L_pt = backend._encode(first_L_mask)
        top_pts: List = [None]
        bot_pts: List = [None]
        for d in range(1, L):
            top_mask = [0.0] * n_slots
            bot_mask = [0.0] * n_slots
            for j in range(n_cols):
                base = j * L
                for i in range(L - d):
                    top_mask[base + i] = 1.0
                for i in range(L - d, L):
                    bot_mask[base + i] = 1.0
            top_pts.append(backend._encode(top_mask))
            bot_pts.append(backend._encode(bot_mask))
        if cache is not None:
            cache[cache_key] = {
                "first_L_pt": first_L_pt,
                "top_pts": top_pts,
                "bot_pts": bot_pts,
            }

    Out: Optional[Ciphertext] = None

    for d in range(L):
        # Step 1: bring row d of each head's A to slots [h*H, h*H + L).
        if d == 0:
            a_row = A_ct
        else:
            a_row = backend.rotate(A_ct, d * L)
        a_row = backend._mul_plain_pt(a_row, first_L_pt)

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
            V_left  = backend.rotate(V_ct, d)
            V_right = backend.rotate(V_ct, d - L)
            V_shift = backend.add(
                backend._mul_plain_pt(V_left, top_pts[d]),
                backend._mul_plain_pt(V_right, bot_pts[d]),
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
# Phase 7e-7: C++-batched column-major linear with cached pre-encoded diagonals
# ---------------------------------------------------------------------------


class ColmajorLinearPlan:
    """Pre-built BSGS plan for ``linear_colmajor_bsgs_cpp``.

    Holds:
      - the BSGS schedule (baby_shifts, giant_shifts, bucket arrays)
      - one pre-encoded :class:`Plaintext` per (giant, baby) term, with
        the giant-shift baked in so the C++ kernel only has to do
        baby-rot, mul_plain, add (no plaintext rotation).
      - the cyclic-replicate steps (depend on backend num_slots and L)
      - bias plaintext (replicated to col-major slots)

    Build cost (one-off): ``in_dim_padded`` plaintext encodes (~30-50 ms each
    on H100 for N=2^16 ⇒ ~30-50 s for hidden=768). After build, every
    inference reuses this plan with zero re-encoding cost.

    All plaintexts are encoded at the SAME chain depth (= depth of input ct
    at execution time). Caller is responsible for invoking ``rebuild`` if
    the input ct depth changes between executions (depth is also recorded
    here so we can detect a mismatch fast).
    """

    def __init__(self, backend, *, L, in_dim, out_dim, in_dim_padded,
                 bs, gs, baby_shifts, giant_shifts, bucket_offsets,
                 bucket_baby_idx, bucket_masks, bias_pt, ct_depth):
        self.backend = backend
        self.L = L
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_padded = in_dim_padded
        self.bs = bs
        self.gs = gs
        self.baby_shifts = baby_shifts
        self.giant_shifts = giant_shifts
        self.bucket_offsets = bucket_offsets
        self.bucket_baby_idx = bucket_baby_idx
        self.bucket_masks = bucket_masks
        self.bias_pt = bias_pt
        self.ct_depth = ct_depth


def build_colmajor_linear_plan(
    backend: CKKSBackend,
    W: np.ndarray,
    *,
    L: int,
    in_dim: int,
    out_dim: int,
    bias: Optional[np.ndarray] = None,
    ct_depth: int = 0,
) -> ColmajorLinearPlan:
    """Pre-encode all BSGS diagonals for one ``linear_colmajor`` matrix.

    The plan is reusable across many invocations on different inputs (same
    W, same chain depth at execution time). Encoded once, queried many.
    """
    n_slots = backend.capabilities.n_slots
    if W.shape != (out_dim, in_dim):
        raise ValueError(f"W.shape {W.shape} != ({out_dim}, {in_dim})")
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

    n = in_dim_padded
    log2_n = n.bit_length() - 1
    bs = 1 << (log2_n // 2)
    gs = n // bs

    # Baby shifts: 0, L, 2L, ..., (bs-1)*L.
    baby_shifts = [i * L for i in range(bs)]
    # Giant shifts: 0, bs*L, 2*bs*L, ..., (gs-1)*bs*L.
    giant_shifts = [j * bs * L for j in range(gs)]

    bucket_offsets = [0]
    bucket_baby_idx = []
    bucket_masks = []

    def _build_diag(d: int) -> np.ndarray:
        v = np.zeros(n_slots, dtype=np.float64)
        for j in range(out_dim):
            col_src = (j + d) % n
            if col_src >= in_dim:
                continue
            w = float(W[j, col_src])
            if w == 0.0:
                continue
            base = j * L
            v[base:base + L] = w
        return v

    def _rot_left_np(v: np.ndarray, k: int) -> np.ndarray:
        k %= n_slots
        if k == 0:
            return v
        return np.concatenate([v[k:], v[:k]])

    for j in range(gs):
        shift = -j * bs * L  # plaintext is left-rotated by `shift`
        for i in range(bs):
            d = i + j * bs
            diag = _build_diag(d)
            if not np.any(diag):
                continue
            diag_shifted = _rot_left_np(diag, shift)
            pt = backend._encode(diag_shifted.tolist())
            # Mod-drop plaintext to ct depth so the C++ multiply_plain_inplace
            # sees matching levels.
            while backend._ops.depth_of_plaintext(pt) < ct_depth:
                backend._ops.mod_drop_inplace_pt(pt)
            bucket_baby_idx.append(i)
            bucket_masks.append(pt)
        bucket_offsets.append(len(bucket_baby_idx))

    bias_vec = None
    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"bias.shape {bias.shape} != ({out_dim},)")
        bvec = np.zeros(n_slots, dtype=np.float64)
        for j in range(out_dim):
            base = j * L
            bvec[base:base + L] = float(bias[j])
        bias_vec = bvec.tolist()

    return ColmajorLinearPlan(
        backend, L=L, in_dim=in_dim, out_dim=out_dim,
        in_dim_padded=in_dim_padded, bs=bs, gs=gs,
        baby_shifts=baby_shifts, giant_shifts=giant_shifts,
        bucket_offsets=bucket_offsets, bucket_baby_idx=bucket_baby_idx,
        bucket_masks=bucket_masks, bias_pt=bias_vec, ct_depth=ct_depth,
    )


# ---------------------------------------------------------------------------
# Phase 8i: streaming column-major linear plan (chunked plaintext encoding)
# ---------------------------------------------------------------------------


class ColmajorLinearPlanStreaming:
    """Memory-frugal variant of :class:`ColmajorLinearPlan`.

    Stores per-giant *raw float arrays* instead of encoded plaintexts.
    At execution time each giant is encoded → multiplied → freed before
    the next giant starts. Peak plaintext memory == bs × ct_size, vs
    (bs·gs) × ct_size for the eager plan.

    For BERT-base hidden=768, bs=gs=32, in_dim_padded=1024 ⇒ 1024 raw
    diagonal vectors are kept on CPU as float64 arrays (~8 KB each at
    the dense slot count after packing zeros — actually
    n_slots·8 B = 524 KB per diag at N=2^17, ~510 MB total for one
    matrix on CPU host RAM). On GPU, peak is 32 plaintexts × ~46 MB =
    ~1.5 GB instead of 47 GB.

    Build cost: ~no encoding (just numpy array allocation + rotate).
    Execution cost: one encode per term per call (no plan reuse), but
    avoids the pool-exceed wall.
    """

    def __init__(self, backend, *, L, in_dim, out_dim, in_dim_padded,
                 bs, gs, baby_shifts, giant_shifts,
                 per_giant_baby_idx, per_giant_diag_arrays,
                 bias_vec, ct_depth):
        self.backend = backend
        self.L = L
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_padded = in_dim_padded
        self.bs = bs
        self.gs = gs
        self.baby_shifts = baby_shifts
        self.giant_shifts = giant_shifts
        # per_giant_baby_idx[g] = list of baby indices used by giant g
        # per_giant_diag_arrays[g] = list of np.ndarray (n_slots,) — diagonals
        # already left-rotated by -g·bs·L so encoder sees the final values.
        self.per_giant_baby_idx = per_giant_baby_idx
        self.per_giant_diag_arrays = per_giant_diag_arrays
        self.bias_vec = bias_vec
        self.ct_depth = ct_depth


def build_colmajor_linear_plan_streaming(
    backend: CKKSBackend,
    W: np.ndarray,
    *,
    L: int,
    in_dim: int,
    out_dim: int,
    bias: Optional[np.ndarray] = None,
    ct_depth: int = 0,
) -> ColmajorLinearPlanStreaming:
    """Build a streaming plan: holds raw diagonals on CPU, encodes JIT."""
    n_slots = backend.capabilities.n_slots
    if W.shape != (out_dim, in_dim):
        raise ValueError(f"W.shape {W.shape} != ({out_dim}, {in_dim})")
    max_cols = n_slots // L
    if max_cols * L != n_slots or (max_cols & (max_cols - 1)) != 0:
        raise ValueError(
            f"L={L} doesn't cleanly divide num_slots={n_slots}"
        )
    in_dim_padded = 1
    while in_dim_padded < in_dim:
        in_dim_padded <<= 1
    if in_dim_padded > max_cols:
        raise ValueError(f"in_dim_padded={in_dim_padded} > max_cols={max_cols}")

    n = in_dim_padded
    log2_n = n.bit_length() - 1
    bs = 1 << (log2_n // 2)
    gs = n // bs
    baby_shifts = [i * L for i in range(bs)]
    giant_shifts = [j * bs * L for j in range(gs)]

    def _build_diag(d: int) -> np.ndarray:
        v = np.zeros(n_slots, dtype=np.float64)
        for j in range(out_dim):
            col_src = (j + d) % n
            if col_src >= in_dim:
                continue
            w = float(W[j, col_src])
            if w == 0.0:
                continue
            base = j * L
            v[base:base + L] = w
        return v

    def _rot_left_np(v: np.ndarray, k: int) -> np.ndarray:
        k %= n_slots
        if k == 0:
            return v
        return np.concatenate([v[k:], v[:k]])

    per_giant_baby_idx: List[List[int]] = []
    per_giant_diag_arrays: List[List[np.ndarray]] = []
    for j in range(gs):
        shift = -j * bs * L
        baby_idx_g: List[int] = []
        diags_g: List[np.ndarray] = []
        for i in range(bs):
            d = i + j * bs
            diag = _build_diag(d)
            if not np.any(diag):
                continue
            diags_g.append(_rot_left_np(diag, shift))
            baby_idx_g.append(i)
        per_giant_baby_idx.append(baby_idx_g)
        per_giant_diag_arrays.append(diags_g)

    bias_vec = None
    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"bias.shape {bias.shape} != ({out_dim},)")
        bvec = np.zeros(n_slots, dtype=np.float64)
        for j in range(out_dim):
            base = j * L
            bvec[base:base + L] = float(bias[j])
        bias_vec = bvec.tolist()

    return ColmajorLinearPlanStreaming(
        backend, L=L, in_dim=in_dim, out_dim=out_dim,
        in_dim_padded=in_dim_padded, bs=bs, gs=gs,
        baby_shifts=baby_shifts, giant_shifts=giant_shifts,
        per_giant_baby_idx=per_giant_baby_idx,
        per_giant_diag_arrays=per_giant_diag_arrays,
        bias_vec=bias_vec, ct_depth=ct_depth,
    )


def linear_colmajor_streaming(
    backend: CKKSBackend,
    x_ct: Ciphertext,
    plan: ColmajorLinearPlanStreaming,
    pt_cache: Optional[dict] = None,
    pt_cache_key=None,
) -> Ciphertext:
    """Execute a streaming column-major linear.

    Plaintexts are encoded one giant at a time, multiplied via the
    C++ ``accumulate_giant`` kernel, freed before the next giant. Peak
    plaintext GPU memory ≈ bs × ct_size, independent of in_dim.

    If ``pt_cache`` is provided, the per-giant encoded plaintext lists
    are stored under ``pt_cache_key`` and reused on subsequent calls
    with the same plan + ct_depth (saves the dominant encoding cost).
    Cached plaintexts are mod-dropped in place to the plan depth on
    first use; the caller must guarantee identical ct_depth on reuse.
    """
    n_slots = backend.capabilities.n_slots
    L = plan.L
    in_dim_padded = plan.in_dim_padded

    # 1. Replicate x cyclically.
    x_replicated = x_ct
    cur = in_dim_padded * L
    while cur < n_slots:
        x_replicated = backend.add(
            x_replicated, backend.rotate(x_replicated, -cur)
        )
        cur <<= 1

    actual_depth = backend._ops.depth(x_replicated)
    if actual_depth != plan.ct_depth:
        raise RuntimeError(
            f"plan was built for ct_depth={plan.ct_depth} but input is at "
            f"depth {actual_depth}"
        )

    # 2. Galois keys.
    needed_keys = set(plan.baby_shifts + plan.giant_shifts) - {0}
    if needed_keys:
        backend.register_rotation_keys(needed_keys)

    # 3. Pre-rotate babies (kept on GPU across all giants — bs cts × ct_size).
    baby_rots = backend._ops.prepare_baby_rotations(
        x_replicated, backend._gk, plan.baby_shifts,
    )

    # 4. Iterate giants: encode bucket → accumulate → free (or cache).
    use_cache = pt_cache is not None and pt_cache_key is not None
    cached_giants = pt_cache.get(pt_cache_key) if use_cache else None
    if use_cache and cached_giants is None:
        # First time we've seen this key: build cache entries we'll keep.
        new_giants: List[List] = []

    Y = None
    for g in range(plan.gs):
        baby_idx = plan.per_giant_baby_idx[g]
        if not baby_idx:
            if use_cache and cached_giants is None:
                new_giants.append([])
            continue

        if cached_giants is not None:
            masks = cached_giants[g]
        else:
            diag_arrays = plan.per_giant_diag_arrays[g]
            masks = []
            for arr in diag_arrays:
                pt = backend._encode(arr.tolist())
                while backend._ops.depth_of_plaintext(pt) < plan.ct_depth:
                    backend._ops.mod_drop_inplace_pt(pt)
                masks.append(pt)
            if use_cache and cached_giants is None:
                new_giants.append(masks)

        giant_ct = backend._ops.accumulate_giant(
            baby_rots, backend._gk, baby_idx, masks, plan.giant_shifts[g],
        )
        if Y is None:
            Y = giant_ct
        else:
            backend._ops.add_inplace(Y, giant_ct)
            del giant_ct

    if use_cache and cached_giants is None:
        pt_cache[pt_cache_key] = new_giants

    if Y is None:
        Y = backend.mul_plain(x_ct, [0.0] * n_slots)

    if plan.bias_vec is not None:
        Y = backend.add_plain(Y, plan.bias_vec)

    return Y


def linear_colmajor_streaming_batched(
    backend: CKKSBackend,
    x_cts: Sequence[Ciphertext],
    plan: "ColmajorLinearPlanStreaming",
) -> List[Ciphertext]:
    """Same as :func:`linear_colmajor_streaming` but applied to N inputs at
    once with the W plaintexts encoded ONCE per giant and shared.

    Cost model: encoding ~ O(gs · bs)  one-time
                accumulate ~ O(N · gs · bs)
    Per-input wall scales as ``encode/N + accumulate``; for our shapes
    (cpc=256, chain=28) the encode share drops to ~5 ms/input at N=32
    while baseline is ~290 ms/input — a 2.5× plateau.

    All inputs MUST be at the same depth ``plan.ct_depth`` and share the
    same Galois key set. The output list is in the same order as ``x_cts``.
    """
    n_slots = backend.capabilities.n_slots
    L = plan.L
    in_dim_padded = plan.in_dim_padded

    # 1. Replicate every input cyclically.
    x_reps: List[Ciphertext] = []
    for x_ct in x_cts:
        x_rep = x_ct
        cur = in_dim_padded * L
        while cur < n_slots:
            x_rep = backend.add(x_rep, backend.rotate(x_rep, -cur))
            cur <<= 1
        x_reps.append(x_rep)

    for x_rep in x_reps:
        actual_depth = backend._ops.depth(x_rep)
        if actual_depth != plan.ct_depth:
            raise RuntimeError(
                f"plan was built for ct_depth={plan.ct_depth} but input is at "
                f"depth {actual_depth}"
            )

    # 2. Galois keys.
    needed_keys = set(plan.baby_shifts + plan.giant_shifts) - {0}
    if needed_keys:
        backend.register_rotation_keys(needed_keys)

    # 3. Pre-rotate babies for each input (kept on GPU across all giants).
    baby_rots_per_input = [
        backend._ops.prepare_baby_rotations(x_rep, backend._gk, plan.baby_shifts)
        for x_rep in x_reps
    ]

    # 4. Iterate giants — encode masks ONCE, apply across all inputs, free.
    Ys: List[Optional[Ciphertext]] = [None] * len(x_cts)
    for g in range(plan.gs):
        baby_idx = plan.per_giant_baby_idx[g]
        if not baby_idx:
            continue

        diag_arrays = plan.per_giant_diag_arrays[g]
        masks = []
        for arr in diag_arrays:
            pt = backend._encode(arr.tolist())
            while backend._ops.depth_of_plaintext(pt) < plan.ct_depth:
                backend._ops.mod_drop_inplace_pt(pt)
            masks.append(pt)

        for i, baby_rots in enumerate(baby_rots_per_input):
            giant_ct = backend._ops.accumulate_giant(
                baby_rots, backend._gk, baby_idx, masks, plan.giant_shifts[g],
            )
            if Ys[i] is None:
                Ys[i] = giant_ct
            else:
                backend._ops.add_inplace(Ys[i], giant_ct)
                del giant_ct

        del masks  # free encoded W before next giant

    # 5. Bias add.
    if plan.bias_vec is not None:
        for i in range(len(Ys)):
            Ys[i] = backend.add_plain(Ys[i], plan.bias_vec)

    # 6. Empty fallback for degenerate plans.
    for i, y in enumerate(Ys):
        if y is None:
            Ys[i] = backend.mul_plain(x_cts[i], [0.0] * n_slots)
    return Ys


def linear_colmajor_multi_streaming_batched(
    backend: CKKSBackend,
    x_cts_per_input: Sequence[Sequence[Ciphertext]],
    W: np.ndarray,
    *,
    L: int,
    in_dim: int,
    out_dim: int,
    bias: Optional[np.ndarray] = None,
) -> List[List[Ciphertext]]:
    """Multi-ct col-major linear with W encoding shared across N inputs.

    ``x_cts_per_input`` is a list of length N, each entry a list of input cts
    for that inference. Output: list of length N, each a list of output cts.

    Memory model: only ONE giant's W plaintexts live on GPU at a time.
    The active baby_rots for ALL N inputs (per (k_in)) live on GPU during
    each (k_out, k_in) sub-block — peak ~ N × bs × ct_size per sub-block.
    """
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_in_cts = (in_dim + cols_per_ct - 1) // cols_per_ct
    n_out_cts = (out_dim + cols_per_ct - 1) // cols_per_ct
    N = len(x_cts_per_input)
    for x_cts in x_cts_per_input:
        if len(x_cts) != n_in_cts:
            raise ValueError(
                f"each input must have {n_in_cts} cts; got {len(x_cts)}"
            )
    if W.shape != (out_dim, in_dim):
        raise ValueError(f"W.shape {W.shape} != ({out_dim}, {in_dim})")

    out: List[List[Optional[Ciphertext]]] = [[None] * n_out_cts for _ in range(N)]
    for k_out in range(n_out_cts):
        out_lo = k_out * cols_per_ct
        out_hi = min(out_lo + cols_per_ct, out_dim)
        out_chunk = out_hi - out_lo
        for k_in in range(n_in_cts):
            in_lo = k_in * cols_per_ct
            in_hi = min(in_lo + cols_per_ct, in_dim)
            in_chunk = in_hi - in_lo
            depth_in = backend._ops.depth(x_cts_per_input[0][k_in])

            W_sub = np.ascontiguousarray(W[out_lo:out_hi, in_lo:in_hi])
            plan = build_colmajor_linear_plan_streaming(
                backend, W_sub, L=L,
                in_dim=in_chunk, out_dim=out_chunk,
                bias=None, ct_depth=depth_in,
            )
            inputs_for_subblock = [x_cts_per_input[i][k_in] for i in range(N)]
            Ys = linear_colmajor_streaming_batched(
                backend, inputs_for_subblock, plan,
            )
            del plan
            for i, Y in enumerate(Ys):
                if out[i][k_out] is None:
                    out[i][k_out] = Y
                else:
                    out[i][k_out] = backend.add(out[i][k_out], Y)

    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"bias.shape {bias.shape} != ({out_dim},)")
        for k_out in range(n_out_cts):
            out_lo = k_out * cols_per_ct
            out_hi = min(out_lo + cols_per_ct, out_dim)
            chunk = out_hi - out_lo
            bvec = [0.0] * n_slots
            for j in range(chunk):
                base = j * L
                bv = float(bias[out_lo + j])
                for i in range(L):
                    bvec[base + i] = bv
            for i in range(N):
                out[i][k_out] = backend.add_plain(out[i][k_out], bvec)

    return [[c for c in row] for row in out]


def linear_colmajor_bsgs_cpp(
    backend: CKKSBackend,
    x_ct: Ciphertext,
    plan: ColmajorLinearPlan,
) -> Ciphertext:
    """Execute a pre-built column-major linear via C++ ``gather_slots_bsgs``.

    Internally:
      1. Replicate x cyclically to fill all num_slots (Python doubling).
      2. Single C++ call: per-giant accumulate (baby_rot · mask), lazy
         rescale, giant rotate, sum across giants.
      3. (Optional) add bias.

    Cost: ``bs + gs - 2`` rotations + ``in_dim_padded`` plain-muls
    (no Python overhead, no per-mul rescale) + ``gs`` giant rotates.

    Depth: +1 (single rescale per giant, all in C++).
    """
    n_slots = backend.capabilities.n_slots
    L = plan.L
    in_dim_padded = plan.in_dim_padded

    # 1. Replicate x to fill num_slots.
    x_replicated = x_ct
    cur = in_dim_padded * L
    while cur < n_slots:
        x_replicated = backend.add(
            x_replicated, backend.rotate(x_replicated, -cur)
        )
        cur <<= 1

    # Sanity: plan was built for a specific input depth.
    actual_depth = backend._ops.depth(x_replicated)
    if actual_depth != plan.ct_depth:
        raise RuntimeError(
            f"plan was built for ct_depth={plan.ct_depth} but input is at "
            f"depth {actual_depth}; rebuild the plan or wrap x in mod-drop"
        )

    # 2. Register Galois keys (no-op if already cached).
    needed_keys = set(plan.baby_shifts + plan.giant_shifts) - {0}
    if needed_keys:
        backend.register_rotation_keys(needed_keys)

    # 3. Single C++ call.
    Y = backend._ops.gather_slots_bsgs(
        x_replicated, backend._gk,
        plan.baby_shifts, plan.giant_shifts,
        plan.bucket_offsets, plan.bucket_baby_idx, plan.bucket_masks,
    )

    # 4. Bias (one-off encrypt + add).
    if plan.bias_pt is not None:
        Y = backend.add_plain(Y, plan.bias_pt)

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


# ---------------------------------------------------------------------------
# Phase 8k: multi-ciphertext col-major (for L=128 hidden=768 at N=2^16)
# ---------------------------------------------------------------------------
#
# Layout: hidden dim of size H is split into n_cts groups of cols_per_ct
# columns each, where cols_per_ct = num_slots // L.
# x_cts[k] holds slot[j*L + i] = X[i, k*cols_per_ct + j]  for j ∈ [0, cols_per_ct).
# The last ct may have fewer than cols_per_ct active columns; pad slots
# are kept zero.


def _cols_per_ct(backend: CKKSBackend, L: int) -> int:
    return backend.capabilities.n_slots // L


def pack_colmajor_multi(
    backend: CKKSBackend,
    X: np.ndarray,
    *,
    L: int,
    hidden: int,
) -> List[Ciphertext]:
    """Encrypt ``X[L, hidden]`` as a list of column-major cts.

    Each ct holds up to ``cols_per_ct = n_slots // L`` columns.
    """
    if X.shape != (L, hidden):
        raise ValueError(f"X.shape {X.shape} != ({L}, {hidden})")
    if (L & (L - 1)) != 0:
        raise ValueError(f"L must be power of 2; got {L}")
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_cts = (hidden + cols_per_ct - 1) // cols_per_ct
    out = []
    for k in range(n_cts):
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, hidden)
        buf = [0.0] * n_slots
        for j in range(lo, hi):
            base = (j - lo) * L
            for i in range(L):
                buf[base + i] = float(X[i, j])
        out.append(backend.encrypt(buf))
    return out


def unpack_colmajor_multi(
    backend: CKKSBackend,
    cts: Sequence[Ciphertext],
    *,
    L: int,
    hidden: int,
) -> np.ndarray:
    cols_per_ct = _cols_per_ct(backend, L)
    out = np.zeros((L, hidden), dtype=np.float64)
    for k, ct in enumerate(cts):
        slots = backend.decrypt(ct)
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, hidden)
        for j in range(lo, hi):
            base = (j - lo) * L
            for i in range(L):
                out[i, j] = slots[base + i]
    return out


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


def layernorm_colmajor_multi(
    backend: CKKSBackend,
    x_cts: Sequence[Ciphertext],
    *,
    L: int,
    hidden: int,
    invsqrt_power_coeffs: Sequence[float],
    invsqrt_interval: tuple,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> List[Ciphertext]:
    """LN over a multi-ct col-major row image."""
    from math import comb
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_cts = len(x_cts)
    h = hidden
    inv_h = 1.0 / h

    a, b = invsqrt_interval
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    n = len(invsqrt_power_coeffs)
    absorbed = [0.0] * n
    for k, c in enumerate(invsqrt_power_coeffs):
        for r in range(k + 1):
            absorbed[r] += float(c) * (sscale ** r) * (sshift ** (k - r)) * comb(k, r)

    # μ broadcast (cross-ct).
    mean_bc = per_col_sum_multi(
        backend, x_cts, L=L, hidden_per_ct=cols_per_ct, scale=inv_h,
    )
    centred = sub_multi(backend, x_cts, mean_bc)

    # Mask off pad cols in the final ct if hidden % cols_per_ct != 0.
    last_active = h - (n_cts - 1) * cols_per_ct
    if last_active < cols_per_ct:
        mask = [0.0] * n_slots
        for j in range(last_active):
            base = j * L
            for i in range(L):
                mask[base + i] = 1.0
        centred[-1] = backend.mul_plain(centred[-1], mask)

    sq = mul_multi(backend, centred, centred)
    var_bc = per_col_sum_multi(
        backend, sq, L=L, hidden_per_ct=cols_per_ct, scale=inv_h,
    )
    inv_sigma = backend.polyval(var_bc[0], absorbed)
    inv_sigma_list = [inv_sigma] * n_cts

    scaled = mul_multi(backend, centred, inv_sigma_list)

    out = []
    for k, sc in enumerate(scaled):
        lo = k * cols_per_ct
        hi = min(lo + cols_per_ct, h)
        chunk = hi - lo
        gv = [0.0] * n_slots
        bv = [0.0] * n_slots
        for j in range(chunk):
            base = j * L
            g = float(gamma[lo + j])
            b_ = float(beta[lo + j])
            for i in range(L):
                gv[base + i] = g
                bv[base + i] = b_
        s = backend.mul_plain(sc, gv)
        out.append(backend.add_plain(s, bv))
    return out


def linear_colmajor_multi_streaming(
    backend: CKKSBackend,
    x_cts: Sequence[Ciphertext],
    W: np.ndarray,
    *,
    L: int,
    in_dim: int,
    out_dim: int,
    bias: Optional[np.ndarray] = None,
    pt_cache: Optional[dict] = None,
    cache_tag: Optional[str] = None,
) -> List[Ciphertext]:
    """Multi-ct col-major linear: ``Y = W @ X`` with X, Y as ct lists.

    For each (k_out, k_in) pair, builds a streaming sub-plan over the
    ``W[k_out_cols, k_in_cols]`` block and accumulates into the output ct.

    If ``pt_cache`` is provided plus a stable ``cache_tag`` (e.g. "Wq"),
    encoded plaintexts are reused across calls. The caller MUST guarantee
    identical ct depths for x_cts on every reuse — typical for a multi-layer
    BERT loop where each layer starts at the same logical depth.
    """
    n_slots = backend.capabilities.n_slots
    cols_per_ct = _cols_per_ct(backend, L)
    n_in_cts = (in_dim + cols_per_ct - 1) // cols_per_ct
    n_out_cts = (out_dim + cols_per_ct - 1) // cols_per_ct
    if len(x_cts) != n_in_cts:
        raise ValueError(
            f"len(x_cts)={len(x_cts)} != expected n_in_cts={n_in_cts}"
        )
    if W.shape != (out_dim, in_dim):
        raise ValueError(f"W.shape {W.shape} != ({out_dim}, {in_dim})")

    out: List[Optional[Ciphertext]] = [None] * n_out_cts
    for k_out in range(n_out_cts):
        out_lo = k_out * cols_per_ct
        out_hi = min(out_lo + cols_per_ct, out_dim)
        out_chunk = out_hi - out_lo
        for k_in in range(n_in_cts):
            in_lo = k_in * cols_per_ct
            in_hi = min(in_lo + cols_per_ct, in_dim)
            in_chunk = in_hi - in_lo
            depth_in = backend._ops.depth(x_cts[k_in])

            cache_key = None
            if pt_cache is not None and cache_tag is not None:
                cache_key = (cache_tag, k_out, k_in, depth_in)

            # Build plan unconditionally (cheap: just numpy ops, no encoding).
            W_sub = np.ascontiguousarray(W[out_lo:out_hi, in_lo:in_hi])
            plan = build_colmajor_linear_plan_streaming(
                backend, W_sub, L=L,
                in_dim=in_chunk, out_dim=out_chunk,
                bias=None, ct_depth=depth_in,
            )
            Y = linear_colmajor_streaming(
                backend, x_cts[k_in], plan,
                pt_cache=pt_cache, pt_cache_key=cache_key,
            )
            del plan
            if out[k_out] is None:
                out[k_out] = Y
            else:
                out[k_out] = backend.add(out[k_out], Y)

    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"bias.shape {bias.shape} != ({out_dim},)")
        for k_out in range(n_out_cts):
            out_lo = k_out * cols_per_ct
            out_hi = min(out_lo + cols_per_ct, out_dim)
            chunk = out_hi - out_lo
            bvec = [0.0] * n_slots
            for j in range(chunk):
                base = j * L
                bv = float(bias[out_lo + j])
                for i in range(L):
                    bvec[base + i] = bv
            out[k_out] = backend.add_plain(out[k_out], bvec)

    return [c for c in out]

