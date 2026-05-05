"""Phase 8k: full L=128 BERT-base layer bench at N=2^16 with multi-ct streaming.

Projects 12-layer e2e cost. Targets <150s. If linear ops dominate,
followup optimizations (plan-cache, fused QKV) can be added.
"""
import gc, time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor_multi,
    qk_scores_nexus, attn_apply_nexus,
    layernorm_colmajor_multi,
    linear_colmajor_multi_streaming,
    add_multi,
    prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 128
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads
    inter = 4 * hidden

    log(f"Config L={L} hidden={hidden} heads={num_heads} head_dim={head_dim}")
    N = 1 << 16
    log(f"Init backend N={N}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 28,
        p_prime_bits=(60, 60),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")

    log("Pre-register rotation keys (max_dim=768 = cols_per_ct=256 max for sub-linears)...")
    t = time.time()
    n_new = prepare_colmajor_keys(be, L=L, max_dim=256)
    log(f"  +{n_new} keys in {time.time()-t:.1f}s")

    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((hidden, hidden)) * 0.05
    Wk = rng.standard_normal((hidden, hidden)) * 0.05
    Wv = rng.standard_normal((hidden, hidden)) * 0.05
    Wo = rng.standard_normal((hidden, hidden)) * 0.05
    W1 = rng.standard_normal((inter, hidden)) * 0.05
    W2 = rng.standard_normal((hidden, inter)) * 0.05
    gamma1, beta1 = np.ones(hidden), np.zeros(hidden)
    gamma2, beta2 = np.ones(hidden), np.zeros(hidden)

    invsqrt_coeffs = [1.0, -0.5, 0.375]
    softmax_coeffs = [1.0, 0.5, 0.125]

    x = rng.standard_normal((L, hidden)) * 0.1
    log("Encrypt input col-major (multi-ct)...")
    x_cts = pack_colmajor_multi(be, x, L=L, hidden=hidden)
    log(f"  {len(x_cts)} input cts")

    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    times = {}

    def stamp(name, dt, depth=None):
        times[name] = times.get(name, 0.0) + dt
        d = f" depth={depth}" if depth is not None else ""
        log(f"  {name:<22s} {dt*1000:8.1f} ms{d}")

    def lin(name, x_in_list, W, in_d, out_d, bias=None):
        t = time.time()
        Y = linear_colmajor_multi_streaming(
            be, x_in_list, W, L=L, in_dim=in_d, out_dim=out_d, bias=bias,
        )
        stamp(name, time.time() - t, depth=be._ops.depth(Y[0]))
        gc.collect()
        return Y

    log("=== ATTENTION ===")
    Q_list = lin("Wq", x_cts, Wq, hidden, hidden)
    K_list = lin("Wk", x_cts, Wk, hidden, hidden)
    V_list = lin("Wv", x_cts, Wv, hidden, hidden)

    # 4 heads per ct (head_dim*L = 64*128 = 8192; 4 heads * 8192 = 32768 = n_slots).
    num_heads_per_ct = 4
    # Per-ct caches: qk/av mutate cached pts via mod-drop, so we cannot share
    # one cache across multiple QKV-ct calls.
    attn_caches = [{} for _ in range(len(Q_list))]
    S_list = []
    t = time.time()
    for k in range(len(Q_list)):
        S_k = qk_scores_nexus(
            be, Q_list[k], K_list[k], L=L, head_dim=head_dim, scale=inv_sqrt_d,
            num_heads_per_ct=num_heads_per_ct, cache=attn_caches[k],
        )
        S_list.append(S_k)
    stamp("qk", time.time() - t, depth=be._ops.depth(S_list[0]))
    del Q_list, K_list; gc.collect()

    t = time.time()
    Asm_list = [be.polyval(S, list(softmax_coeffs)) for S in S_list]
    stamp("softmax", time.time() - t, depth=be._ops.depth(Asm_list[0]))
    del S_list; gc.collect()

    t = time.time()
    O_list = []
    for k in range(len(V_list)):
        O_k = attn_apply_nexus(
            be, Asm_list[k], V_list[k], L=L, head_dim=head_dim,
            num_heads_per_ct=num_heads_per_ct, cache=attn_caches[k],
        )
        O_list.append(O_k)
    stamp("av", time.time() - t, depth=be._ops.depth(O_list[0]))
    del V_list, Asm_list; gc.collect()

    log("=== Wo + LN1 ===")
    Wo_out = lin("Wo", O_list, Wo, hidden, hidden)
    del O_list; gc.collect()
    t = time.time()
    res1 = add_multi(be, Wo_out, x_cts)
    stamp("residual1", time.time() - t)
    del Wo_out; gc.collect()
    t = time.time()
    h1 = layernorm_colmajor_multi(
        be, res1, L=L, hidden=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma1, beta=beta1,
    )
    stamp("LN1", time.time() - t, depth=be._ops.depth(h1[0]))
    del res1; gc.collect()

    log("=== FFN ===")
    H = lin("W1", h1, W1, hidden, inter)
    t = time.time()
    G = [be.polyval(c, [0.5, 0.398, 0.0, -0.0066]) for c in H]
    stamp("gelu", time.time() - t, depth=be._ops.depth(G[0]))
    del H; gc.collect()

    H2 = lin("W2", G, W2, inter, hidden)
    del G; gc.collect()

    log("=== LN2 ===")
    t = time.time()
    res2 = add_multi(be, H2, h1)
    stamp("residual2", time.time() - t)
    del H2, h1; gc.collect()
    t = time.time()
    out = layernorm_colmajor_multi(
        be, res2, L=L, hidden=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma2, beta=beta2,
    )
    stamp("LN2", time.time() - t, depth=be._ops.depth(out[0]))

    print()
    print("=" * 70)
    print(f"L={L} MULTI-CT per-layer  hidden={hidden}  N={N}")
    print("=" * 70)
    total = sum(times.values())
    for name, dt in sorted(times.items(), key=lambda kv: -kv[1]):
        if total > 0:
            print(f"  {name:<22s}  {dt*1000:9.1f} ms   ({100*dt/total:5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL (1 layer)':<22s}  {total*1000:9.1f} ms")
    print(f"  {'PROJECTED 12-layer':<22s}  {total*12:9.1f} s   "
          f"(NEXUS L=128: 37.3s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
