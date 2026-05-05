"""1-layer bench at tight chain to reveal per-op breakdown.

Drops chain to 20 (one layer fits cleanly: ~17 levels needed).
Skips the layer-2 cold/warm comparison — instead profiles layer 1 only with
a fresh cache, then reruns just the linears alone with the warm cache to
show the W-encoding contribution.
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
    chain_levels = 28    # original; OOMs only when pt_cache enabled
    log(f"Init backend N={N} chain_levels={chain_levels}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * chain_levels,
        p_prime_bits=(60,),
        scale_bits=50,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be._num_slots}")
    n_new = prepare_colmajor_keys(be, L=L, max_dim=256)
    log(f"  +{n_new} keys")

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
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)

    # pt_cache disabled: at N=2^16 chain=20 the encoded-W plaintexts blow past
    # RMM's 78 GiB cap. Re-encode each call (the cost we will profile).
    pt_cache = None
    attn_caches = [{} for _ in range(3)]

    def run_layer(x_cts, layer_idx, label):
        layer_t = time.time()
        times = {}
        def stamp(name, dt):
            times[name] = times.get(name, 0.0) + dt

        for tag, W in [("Wq", Wq), ("Wk", Wk), ("Wv", Wv)]:
            t = time.time()
            Y = linear_colmajor_multi_streaming(
                be, x_cts, W, L=L, in_dim=hidden, out_dim=hidden,
                pt_cache=pt_cache, cache_tag=None,
            )
            stamp(tag, time.time() - t)
            if tag == "Wq": Q = Y
            elif tag == "Wk": K = Y
            else: V = Y

        t = time.time()
        S = [qk_scores_nexus(
                be, Q[k], K[k], L=L, head_dim=head_dim, scale=inv_sqrt_d,
                num_heads_per_ct=4, cache=attn_caches[k],
            ) for k in range(len(Q))]
        stamp("qk", time.time() - t)
        del Q, K; gc.collect()

        t = time.time()
        Asm = [be.polyval(s, list(softmax_coeffs)) for s in S]
        stamp("softmax", time.time() - t)
        del S; gc.collect()

        t = time.time()
        O = [attn_apply_nexus(
                be, Asm[k], V[k], L=L, head_dim=head_dim,
                num_heads_per_ct=4, cache=attn_caches[k],
            ) for k in range(len(V))]
        stamp("av", time.time() - t)
        del V, Asm; gc.collect()

        t = time.time()
        Wo_out = linear_colmajor_multi_streaming(
            be, O, Wo, L=L, in_dim=hidden, out_dim=hidden,
            pt_cache=pt_cache, cache_tag=None,
        )
        stamp("Wo", time.time() - t)
        del O; gc.collect()

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
        stamp("LN1", time.time() - t)
        del res1; gc.collect()

        t = time.time()
        H = linear_colmajor_multi_streaming(
            be, h1, W1, L=L, in_dim=hidden, out_dim=inter,
            pt_cache=pt_cache, cache_tag=None,
        )
        stamp("W1", time.time() - t)

        t = time.time()
        G = [be.polyval(c, [0.5, 0.398, 0.0, -0.0066]) for c in H]
        stamp("gelu", time.time() - t)
        del H; gc.collect()

        t = time.time()
        H2 = linear_colmajor_multi_streaming(
            be, G, W2, L=L, in_dim=inter, out_dim=hidden,
            pt_cache=pt_cache, cache_tag=None,
        )
        stamp("W2", time.time() - t)
        del G; gc.collect()

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
        stamp("LN2", time.time() - t)

        layer_dt = time.time() - layer_t
        log(f"--- {label} TOTAL: {layer_dt:.2f}s ---")
        for k, v in sorted(times.items(), key=lambda kv: -kv[1]):
            log(f"     {k:<14s} {v*1000:8.0f} ms")
        linear_t = sum(times.get(k, 0) for k in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"])
        nonlin_t = layer_dt - linear_t
        log(f"     -> linears={linear_t:.2f}s  non-linears={nonlin_t:.2f}s")
        return out, layer_dt, dict(times)

    x = rng.standard_normal((L, hidden)) * 0.1
    log("Pack input...")
    x_cts = pack_colmajor_multi(be, x, L=L, hidden=hidden)
    log(f"  {len(x_cts)} input cts at depth={be._ops.depth(x_cts[0])}")

    log("=== LAYER (cold cache) ===")
    h_cts, t1, times_cold = run_layer(x_cts, 1, "LAYER cold")

    print()
    print("=" * 72)
    print(f"L={L} 1-LAYER COLD-CACHE  hidden={hidden}  N={N}  chain={chain_levels}")
    print(f"  Total: {t1:.2f}s")
    linear_t = sum(times_cold.get(k, 0) for k in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"])
    nonlin_t = t1 - linear_t
    print(f"  Linears: {linear_t:.2f}s  ({linear_t/t1*100:.0f}%)")
    print(f"  Non-linear (LN/softmax/GELU/attn): {nonlin_t:.2f}s ({nonlin_t/t1*100:.0f}%)")
    print(f"  12-layer projection (cold every layer): {12*t1:.0f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
