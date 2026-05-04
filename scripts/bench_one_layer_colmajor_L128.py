"""Phase 8h: scale Phase 7e column-major bench from L=32 to L=128.

Same kernels — just a larger ring (N=2^17 to fit L*hidden=98304 slots).
Goal: measure honest L=128 cost vs NEXUS reported 37.3s.
"""
import gc
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor,
    linear_colmajor, layernorm_colmajor,
    qk_scores_nexus, attn_apply_nexus,
    prepare_colmajor_keys,
    build_colmajor_linear_plan, linear_colmajor_bsgs_cpp,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 128
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads     # 64
    inter = 4 * hidden                  # 3072

    log(f"Config L={L} hidden={hidden} heads={num_heads} head_dim={head_dim}")
    # n_slots = N/2; need n_slots >= L*hidden = 98304 -> N=2^18 (131072 slots).
    N = 1 << 18
    log(f"Init backend N={N}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        # Tighter chain to fit Galois keys in GPU memory at N=2^18.
        # 1 layer needs ~12 levels (3 linears + softmax + LN + GELU each ~2).
        q_prime_bits=(60,) + (50,) * 14,
        p_prime_bits=(60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be.capabilities.n_slots}")

    # max_dim covers all linear_colmajor + LN rotations.
    log("Pre-register rotation keys (max_dim=2048)...")
    t = time.time()
    n_new = prepare_colmajor_keys(be, L=L, max_dim=2048)
    log(f"  +{n_new} keys in {time.time()-t:.1f}s")

    rng = np.random.default_rng(0)
    Wq_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    Wk_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    Wv_h = rng.standard_normal((num_heads, head_dim, hidden)) * 0.05
    Wq = Wq_h.reshape(num_heads * head_dim, hidden)
    Wk = Wk_h.reshape(num_heads * head_dim, hidden)
    Wv = Wv_h.reshape(num_heads * head_dim, hidden)
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    W1 = rng.standard_normal((inter, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, inter)) * 0.05
    gamma1, beta1 = np.ones(hidden), np.zeros(hidden)
    gamma2, beta2 = np.ones(hidden), np.zeros(hidden)

    invsqrt_coeffs = [1.0, -0.5, 0.375]
    softmax_coeffs = [1.0, 0.5, 0.125]

    x = rng.standard_normal((L, hidden)) * 0.1
    log("Encrypt input col-major...")
    x_ct = pack_colmajor(be, x, L=L, head_dim=hidden)

    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    times = {}
    build_times = {}

    def stamp(name, dt, depth=None):
        times[name] = times.get(name, 0.0) + dt
        d = f" depth={depth}" if depth is not None else ""
        log(f"  {name:<22s} {dt*1000:8.1f} ms{d}")

    def stamp_build(name, dt):
        build_times[name] = build_times.get(name, 0.0) + dt

    def linear_cpp(name, input_ct, W, bias=None):
        in_d = W.shape[1]
        out_d = W.shape[0]
        depth = be._ops.depth(input_ct)
        t = time.time()
        plan = build_colmajor_linear_plan(
            be, W, L=L, in_dim=in_d, out_dim=out_d, bias=bias, ct_depth=depth,
        )
        stamp_build(name, time.time() - t)
        t = time.time()
        Y = linear_colmajor_bsgs_cpp(be, input_ct, plan)
        stamp(name, time.time() - t, depth=be._ops.depth(Y))
        return Y

    log("=== ATTENTION ===")
    gc.collect()
    Q_full = linear_cpp("Wq_fused", x_ct, Wq)
    gc.collect()
    K_full = linear_cpp("Wk_fused", x_ct, Wk)
    gc.collect()
    V_full = linear_cpp("Wv_fused", x_ct, Wv)

    # 12 heads * 64 head_dim * 128 L = 98304 slots, fits in N=131072.
    num_heads_per_ct = 12
    n_groups = num_heads // num_heads_per_ct  # 1
    head_block = head_dim * L
    group_block = num_heads_per_ct * head_block

    O_full = None
    attn_cache: dict = {}
    for g in range(n_groups):
        gc.collect()
        if g == 0:
            Qg, Kg, Vg = Q_full, K_full, V_full
        else:
            t = time.time()
            shift = g * group_block
            Qg = be.rotate(Q_full, shift)
            Kg = be.rotate(K_full, shift)
            Vg = be.rotate(V_full, shift)
            stamp(f"extract[g{g}]", time.time() - t)

        t = time.time()
        Sg = qk_scores_nexus(be, Qg, Kg, L=L, head_dim=head_dim,
                             scale=inv_sqrt_d,
                             num_heads_per_ct=num_heads_per_ct,
                             cache=attn_cache)
        stamp(f"qk[g{g}]", time.time() - t)

        t = time.time()
        Asm = be.polyval(Sg, list(softmax_coeffs))
        stamp(f"softmax[g{g}]", time.time() - t)

        t = time.time()
        Og = attn_apply_nexus(be, Asm, Vg, L=L, head_dim=head_dim,
                              num_heads_per_ct=num_heads_per_ct,
                              cache=attn_cache)
        stamp(f"av[g{g}]", time.time() - t)

        t = time.time()
        if g == 0:
            O_full = Og
        else:
            Og_shifted = be.rotate(Og, -g * group_block)
            O_full = be.add(O_full, Og_shifted)
        stamp(f"scatter[g{g}]", time.time() - t)

    log("=== Wo + LN1 ===")
    O = linear_cpp("Wo", O_full, Wo)
    t = time.time(); res1 = be.add(O, x_ct); stamp("residual1", time.time() - t)
    t = time.time()
    h1 = layernorm_colmajor(
        be, res1, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma1, beta=beta1,
    )
    stamp("LN1", time.time() - t, depth=be._ops.depth(h1))

    log("=== FFN ===")
    slabs = []
    for s in range(4):
        gc.collect()
        Ws = W1[s * hidden:(s + 1) * hidden, :]
        Hs = linear_cpp(f"W1[s{s}]", h1, Ws)
        t = time.time()
        Gs = be.polyval(Hs, [0.5, 0.398, 0.0, -0.0066])
        stamp(f"gelu[s{s}]", time.time() - t, depth=be._ops.depth(Gs))
        slabs.append(Gs)

    log("=== W2 ===")
    H2 = None
    for s in range(4):
        gc.collect()
        Ws2 = W2[:, s * hidden:(s + 1) * hidden]
        Y = linear_cpp(f"W2[s{s}]", slabs[s], Ws2)
        H2 = Y if H2 is None else be.add(H2, Y)

    log("=== LN2 ===")
    t = time.time(); res2 = be.add(H2, h1); stamp("residual2", time.time() - t)
    t = time.time()
    out = layernorm_colmajor(
        be, res2, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma2, beta=beta2,
    )
    stamp("LN2", time.time() - t, depth=be._ops.depth(out))

    print()
    print("=" * 70)
    print(f"COL-MAJOR PER-LAYER BENCH  (L={L} hidden={hidden} heads={num_heads})  N={N}")
    print("=" * 70)
    total = sum(times.values())
    groups = {
        "Wq/Wk/Wv (3 fused)":   times.get("Wq_fused", 0.0) + times.get("Wk_fused", 0.0) + times.get("Wv_fused", 0.0),
        "extract":              sum(v for k, v in times.items() if k.startswith("extract[")),
        "qk":                   sum(v for k, v in times.items() if k.startswith("qk[")),
        "softmax":              sum(v for k, v in times.items() if k.startswith("softmax[")),
        "av":                   sum(v for k, v in times.items() if k.startswith("av[")),
        "scatter":              sum(v for k, v in times.items() if k.startswith("scatter[")),
        "Wo":                   times.get("Wo", 0.0),
        "LN1":                  times.get("LN1", 0.0),
        "W1 (4 slabs)":         sum(v for k, v in times.items() if k.startswith("W1[")),
        "GELU (4 slabs)":       sum(v for k, v in times.items() if k.startswith("gelu[")),
        "W2 (4 slabs)":         sum(v for k, v in times.items() if k.startswith("W2[")),
        "LN2":                  times.get("LN2", 0.0),
        "residual":             times.get("residual1", 0.0) + times.get("residual2", 0.0),
    }
    for name, dt in groups.items():
        if total > 0:
            print(f"  {name:<32s}  {dt*1000:8.1f} ms   ({100*dt/total:5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL (1 layer)':<32s}  {total*1000:8.1f} ms")
    print(f"  {'PROJECTED 12-layer e2e':<32s}  {total*12:8.1f} s   (NEXUS: 37.3s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
