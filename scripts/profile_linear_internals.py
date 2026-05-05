"""Microprofile linear_colmajor_streaming: encode vs rotate vs accumulate.

Run ONE Wq-style linear (256x256 sub-block at depth 0) and time each stage.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor,
    prepare_colmajor_keys,
    build_colmajor_linear_plan_streaming,
)


def main():
    L = 128
    cpc = 256          # cols per ct
    N = 1 << 16
    chain = 28
    print(f"N={N} chain={chain} L={L} cpc={cpc}")

    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * chain,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    prepare_colmajor_keys(be, L=L, max_dim=cpc)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((L, cpc)) * 0.1
    W = rng.standard_normal((cpc, cpc)) * 0.05
    x_ct = pack_colmajor(be, X, L=L, head_dim=cpc)

    plan = build_colmajor_linear_plan_streaming(
        be, W, L=L, in_dim=cpc, out_dim=cpc, bias=None, ct_depth=0,
    )
    print(f"plan: bs={plan.bs}, gs={plan.gs}, in_dim_padded={plan.in_dim_padded}")
    print(f"  baby_shifts ({len(plan.baby_shifts)}): {plan.baby_shifts[:8]}...")
    print(f"  giant_shifts ({len(plan.giant_shifts)}): {plan.giant_shifts[:8]}...")

    # ---- profile each stage ----
    n_slots = be.capabilities.n_slots
    in_dim_padded = plan.in_dim_padded

    # 1. Replicate
    t = time.time()
    x_rep = x_ct
    cur = in_dim_padded * L
    n_rep = 0
    while cur < n_slots:
        x_rep = be.add(x_rep, be.rotate(x_rep, -cur))
        cur <<= 1
        n_rep += 1
    t_rep = time.time() - t
    print(f"\nReplicate ({n_rep} doublings): {t_rep*1000:.1f}ms")

    # 2. baby rotations
    needed = set(plan.baby_shifts + plan.giant_shifts) - {0}
    be.register_rotation_keys(needed)
    t = time.time()
    baby_rots = be._ops.prepare_baby_rotations(x_rep, be._gk, plan.baby_shifts)
    t_baby = time.time() - t
    print(f"prepare_baby_rotations ({plan.bs} rots): {t_baby*1000:.1f}ms ({t_baby*1000/max(plan.bs,1):.2f}ms/rot)")

    # 3. Encode + accumulate per giant
    t_encode_total = 0.0
    t_accum_total = 0.0
    n_pts_encoded = 0
    Y = None
    for g in range(plan.gs):
        baby_idx = plan.per_giant_baby_idx[g]
        if not baby_idx:
            continue

        # Encode this giant's masks
        diag_arrays = plan.per_giant_diag_arrays[g]
        t_e = time.time()
        masks = []
        for arr in diag_arrays:
            pt = be._encode(arr.tolist())
            while be._ops.depth_of_plaintext(pt) < plan.ct_depth:
                be._ops.mod_drop_inplace_pt(pt)
            masks.append(pt)
        t_encode_total += time.time() - t_e
        n_pts_encoded += len(masks)

        # Accumulate
        t_a = time.time()
        giant_ct = be._ops.accumulate_giant(
            baby_rots, be._gk, baby_idx, masks, plan.giant_shifts[g],
        )
        t_accum_total += time.time() - t_a

        if Y is None:
            Y = giant_ct
        else:
            be._ops.add_inplace(Y, giant_ct)

    print(f"Encode pts ({n_pts_encoded} total): {t_encode_total*1000:.1f}ms ({t_encode_total*1000/max(n_pts_encoded,1):.2f}ms/pt)")
    print(f"Accumulate giants ({plan.gs} giants): {t_accum_total*1000:.1f}ms ({t_accum_total*1000/max(plan.gs,1):.2f}ms/giant)")

    total = t_rep + t_baby + t_encode_total + t_accum_total
    print(f"\nTotal: {total*1000:.1f}ms")
    print(f"  Replicate    : {t_rep*1000:7.1f}ms ({t_rep/total*100:5.1f}%)")
    print(f"  Baby rots    : {t_baby*1000:7.1f}ms ({t_baby/total*100:5.1f}%)")
    print(f"  Encode pts   : {t_encode_total*1000:7.1f}ms ({t_encode_total/total*100:5.1f}%)")
    print(f"  Accum giants : {t_accum_total*1000:7.1f}ms ({t_accum_total/total*100:5.1f}%)")
    print()
    print(f"Wq is 9 of these sub-blocks (3x3). Projected Wq cost: {total*9:.2f}s")


if __name__ == "__main__":
    main()
