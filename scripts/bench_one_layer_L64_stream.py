"""Phase 8h-stream: L=64 BERT-base column-major bench using STREAMING
linear_colmajor (no pre-encoded plan).

Trades faster per-call (plan reuse) for memory feasibility — at N=2^17
with depth-22 chain, pre-encoding 1024 plaintexts per matrix needs ~46 GB
which exceeds the HEonGPU pool. The streaming path encodes each diagonal
just-in-time, multiplies, frees — peak memory ~constant.
"""
import gc, time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.ops_attention_nexus import (
    pack_colmajor,
    linear_colmajor, layernorm_colmajor,
    qk_scores_nexus, attn_apply_nexus,
    prepare_colmajor_keys,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    L = 64
    hidden = 768
    num_heads = 12
    head_dim = hidden // num_heads
    inter = 4 * hidden

    log(f"Config L={L} hidden={hidden} heads={num_heads} head_dim={head_dim}")
    N = 1 << 17
    log(f"Init backend N={N}...")
    be = HEonGPUBackend(
        poly_modulus_degree=N,
        q_prime_bits=(60,) + (50,) * 22,
        p_prime_bits=(60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    log(f"  ready max_depth={be._max_depth} n_slots={be.capabilities.n_slots}")

    log("Pre-register rotation keys (max_dim=2048)...")
    t = time.time()
    n_new = prepare_colmajor_keys(be, L=L, max_dim=2048)
    log(f"  +{n_new} keys in {time.time()-t:.1f}s")

    rng = np.random.default_rng(0)
    Wq = (rng.standard_normal((num_heads * head_dim, hidden)) * 0.05)
    Wk = (rng.standard_normal((num_heads * head_dim, hidden)) * 0.05)
    Wv = (rng.standard_normal((num_heads * head_dim, hidden)) * 0.05)
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

    def stamp(name, dt, depth=None):
        times[name] = times.get(name, 0.0) + dt
        d = f" depth={depth}" if depth is not None else ""
        log(f"  {name:<22s} {dt*1000:8.1f} ms{d}")

    def lin(name, x_in, W, bias=None):
        in_d, out_d = W.shape[1], W.shape[0]
        t = time.time()
        Y = linear_colmajor(be, x_in, W, L=L, in_dim=in_d, out_dim=out_d,
                            bias=bias, bsgs=True)
        stamp(name, time.time() - t, depth=be._ops.depth(Y))
        gc.collect()
        return Y

    log("=== ATTENTION ===")
    Q = lin("Wq_fused", x_ct, Wq)
    K = lin("Wk_fused", x_ct, Wk)
    V = lin("Wv_fused", x_ct, Wv)

    # 12 heads * 64 head_dim * 64 L = 49152 slots, fits in N=131072 (n_slots=65536).
    num_heads_per_ct = 12
    attn_cache = {}

    t = time.time()
    S = qk_scores_nexus(be, Q, K, L=L, head_dim=head_dim, scale=inv_sqrt_d,
                         num_heads_per_ct=num_heads_per_ct, cache=attn_cache)
    stamp("qk", time.time() - t, depth=be._ops.depth(S))
    gc.collect()

    t = time.time()
    Asm = be.polyval(S, list(softmax_coeffs))
    stamp("softmax", time.time() - t, depth=be._ops.depth(Asm))
    gc.collect()

    t = time.time()
    O_full = attn_apply_nexus(be, Asm, V, L=L, head_dim=head_dim,
                               num_heads_per_ct=num_heads_per_ct, cache=attn_cache)
    stamp("av", time.time() - t, depth=be._ops.depth(O_full))
    gc.collect()
    del Q, K, V, S, Asm
    gc.collect()

    log("=== Wo + LN1 ===")
    O = lin("Wo", O_full, Wo)
    del O_full; gc.collect()

    t = time.time(); res1 = be.add(O, x_ct); stamp("residual1", time.time() - t)
    del O; gc.collect()

    t = time.time()
    h1 = layernorm_colmajor(
        be, res1, L=L, hidden_dim=hidden,
        invsqrt_power_coeffs=invsqrt_coeffs,
        invsqrt_interval=(-1.0, 1.0),
        gamma=gamma1, beta=beta1,
    )
    stamp("LN1", time.time() - t, depth=be._ops.depth(h1))
    del res1; gc.collect()

    log("=== FFN ===")
    slabs = []
    for s in range(4):
        Ws = W1[s * hidden:(s + 1) * hidden, :]
        Hs = lin(f"W1[s{s}]", h1, Ws)
        t = time.time()
        Gs = be.polyval(Hs, [0.5, 0.398, 0.0, -0.0066])
        stamp(f"gelu[s{s}]", time.time() - t, depth=be._ops.depth(Gs))
        del Hs; gc.collect()
        slabs.append(Gs)

    log("=== W2 ===")
    H2 = None
    for s in range(4):
        Ws2 = W2[:, s * hidden:(s + 1) * hidden]
        Y = lin(f"W2[s{s}]", slabs[s], Ws2)
        H2 = Y if H2 is None else be.add(H2, Y)
        slabs[s] = None; gc.collect()

    log("=== LN2 ===")
    t = time.time(); res2 = be.add(H2, h1); stamp("residual2", time.time() - t)
    del H2, h1; gc.collect()

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
    print(f"COL-MAJOR PER-LAYER (STREAMING)  L={L} hidden={hidden}  N={N}")
    print("=" * 70)
    total = sum(times.values())
    groups = {
        "Wq/Wk/Wv":   sum(times.get(k, 0.0) for k in ("Wq_fused","Wk_fused","Wv_fused")),
        "qk":         times.get("qk", 0.0),
        "softmax":    times.get("softmax", 0.0),
        "av":         times.get("av", 0.0),
        "Wo":         times.get("Wo", 0.0),
        "LN1":        times.get("LN1", 0.0),
        "W1 (4 slabs)": sum(v for k, v in times.items() if k.startswith("W1[")),
        "GELU (4 slabs)": sum(v for k, v in times.items() if k.startswith("gelu[")),
        "W2 (4 slabs)": sum(v for k, v in times.items() if k.startswith("W2[")),
        "LN2":        times.get("LN2", 0.0),
        "residual":   times.get("residual1", 0.0) + times.get("residual2", 0.0),
    }
    for name, dt in groups.items():
        if total > 0:
            print(f"  {name:<24s}  {dt*1000:8.1f} ms   ({100*dt/total:5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL (1 layer)':<24s}  {total*1000:8.1f} ms")
    print(f"  {'PROJECTED 12-layer e2e':<24s}  {total*12:8.1f} s   (NEXUS L=128: 37.3s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
