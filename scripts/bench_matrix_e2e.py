"""End-to-end benchmark of encrypt_inference_matrix on synthetic weights.

Goal: measure where we actually stand (vs the 16h projection from per-token).
Tests BERT-tiny and BERT-base shapes; uses fake weights + dummy LPAN coeffs.
"""
import time
import numpy as np

from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.protocol import (
    LayerWeights, ModelWeights, encrypt_inference_matrix,
)
from fhe_thesis.encryption.coefficients import PolyCoeffs


def make_weights(num_layers: int, hidden: int, num_heads: int, ffn_mult: int = 4):
    layers = []
    rng = np.random.default_rng(0)
    for _ in range(num_layers):
        layers.append(LayerWeights(
            Wq=rng.standard_normal((hidden, hidden)) * 0.02,
            bq=np.zeros(hidden),
            Wk=rng.standard_normal((hidden, hidden)) * 0.02,
            bk=np.zeros(hidden),
            Wv=rng.standard_normal((hidden, hidden)) * 0.02,
            bv=np.zeros(hidden),
            Wo=rng.standard_normal((hidden, hidden)) * 0.02,
            bo=np.zeros(hidden),
            W1=rng.standard_normal((ffn_mult*hidden, hidden)) * 0.02,
            b1=np.zeros(ffn_mult*hidden),
            W2=rng.standard_normal((hidden, ffn_mult*hidden)) * 0.02,
            b2=np.zeros(hidden),
            ln1_gamma=np.ones(hidden),
            ln1_beta=np.zeros(hidden),
            ln2_gamma=np.ones(hidden),
            ln2_beta=np.zeros(hidden),
        ))
    return ModelWeights(
        model_key="synth", num_layers=num_layers,
        hidden=hidden, num_heads=num_heads, layers=layers,
    )


def make_coeffs(num_layers: int):
    # Simple low-degree power-basis polys covering [-1,1] (we won't validate
    # accuracy here, just measure FHE op cost).
    gelu_p = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # ~identity-ish
    sm_p   = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
    invs   = np.array([1.0, -0.5, 0.375, -0.3125, 0.273, -0.246])
    ln_p   = invs
    coeffs = {}
    for i in range(num_layers):
        coeffs[i] = {
            "GELU":    PolyCoeffs(power_coeffs=gelu_p, interval=(-4.0, 4.0), degree=len(gelu_p)-1),
            "Softmax": PolyCoeffs(power_coeffs=sm_p,   interval=(-8.0, 8.0), degree=len(sm_p)-1),
            "LN":      PolyCoeffs(power_coeffs=ln_p,   interval=(0.01, 4.0), degree=len(ln_p)-1),
        }
    return coeffs


def bench(name, num_layers, hidden, num_heads, seq_len):
    print(f"\n==== {name}: L={num_layers}, hidden={hidden}, heads={num_heads}, seq={seq_len} ====")
    backend = HEonGPUBackend(
        poly_modulus_degree=1 << 16,
        q_prime_bits=(60,) + (50,) * 30,
        p_prime_bits=(60, 60, 60),
        scale_bits=50,
        bootstrap_hamming_weight=16,
        sec_none=True,
    )
    backend.configure_bootstrapping()
    weights = make_weights(num_layers, hidden, num_heads)
    coeffs = make_coeffs(num_layers)
    x = np.random.default_rng(1).standard_normal((seq_len, hidden)) * 0.1

    # warm: just run 1 layer worth via direct encrypt+decrypt
    t0 = time.time()
    try:
        logits, timings = encrypt_inference_matrix(
            backend, x, weights, coeffs, block=0,
        )
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return
    wall = time.time() - t0
    print(f"  TOTAL wall: {wall:.2f}s")
    print(f"  pack: tokens_per_ct={int(timings.get('pack.tokens_per_ct',0))} "
          f"num_cts={int(timings.get('pack.num_cts',0))} "
          f"block={int(timings.get('pack.block',0))}")
    # group by stage
    groups = {}
    for k, v in timings.items():
        if not k.startswith("L"):
            continue
        parts = k.split(".", 2)
        if len(parts) < 3:
            continue
        stage = f"{parts[1]}.{parts[2]}"
        groups[stage] = groups.get(stage, 0.0) + v
    print("  Per-stage totals (summed across all layers):")
    for stage, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"    {stage:25s} {t:7.2f}s")


if __name__ == "__main__":
    # BERT-tiny first (fast feedback)
    bench("BERT-tiny",  num_layers=2, hidden=128, num_heads=2,  seq_len=8)
    # BERT-base seq=8 (small but realistic shapes)
    bench("BERT-base s8",  num_layers=12, hidden=768, num_heads=12, seq_len=8)
    # BERT-base seq=64
    bench("BERT-base s64", num_layers=12, hidden=768, num_heads=12, seq_len=64)
