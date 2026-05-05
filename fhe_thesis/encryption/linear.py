"""Column-major linear projections (single-ct + streaming + multi)

Sliced from the original ``ops_attention_nexus.py`` during the production
re-modularization (synthesizer-lpan-production branch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import CKKSBackend, Ciphertext
from .colmajor import pack_colmajor, _cols_per_ct  # noqa: F401

# -------------------------------------------------------------------------
# Column-major linear projections (single-ct + streaming + multi)
# -------------------------------------------------------------------------



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

