"""Run decompress round-by-round, decoding each intermediate to check correctness."""
import math
import numpy as np
from fhe_thesis.encryption.heongpu_backend import HEonGPUBackend
from fhe_thesis.encryption.mm_nexus import required_galois_elts, enc_compress

N_LOG = 12
N = 1 << N_LOG

be = HEonGPUBackend(
    poly_modulus_degree=N,
    q_prime_bits=(60, 50, 50, 50),
    p_prime_bits=(60, 60, 60),
    scale_bits=50,
    sec_none=True,
)
elts = required_galois_elts(N)
gk = be._kg.generate_galois_key_elts(be._ctx, be._sk, elts)

rng = np.random.default_rng(2)
values = rng.normal(size=N).astype(np.float64) * 0.5

ct0 = enc_compress(be, values)

def decode_coeffs(ct):
    pt = be._decryptor.decrypt(be._ctx, ct)
    return np.array(be._encoder.decode_coeff(pt)[:N])

ops = be._ops
temp = [ct0]

for i in range(N_LOG):
    galois_elt = elts[i]
    index_raw = (N << 1) - (1 << i)
    index = (index_raw * galois_elt) % (N << 1)
    new_temp = [None] * (len(temp) << 1)
    for a, t_a in enumerate(temp):
        rotated = ops.apply_galois_elt(t_a, gk, galois_elt)
        sum_ct = be._clone(t_a)
        ops.add_inplace(sum_ct, rotated)
        new_temp[a] = sum_ct
        shifted = be._clone(t_a)
        ops.multiply_power_of_x_inplace(shifted, index_raw)
        rs = be._clone(rotated)
        ops.multiply_power_of_x_inplace(rs, index)
        ops.add_inplace(shifted, rs)
        new_temp[a + len(temp)] = shifted
    temp = new_temp
    # decode the FIRST piece and report stats
    coeffs0 = decode_coeffs(temp[0])
    nonzero = np.where(np.abs(coeffs0) > 1e-3)[0]
    print(f"round {i}: {len(temp)} pieces. piece[0] nonzero positions count={len(nonzero)}, first 6 positions={nonzero[:6].tolist()}, vals={coeffs0[nonzero[:4]].round(4).tolist()}")
    # sanity: piece[0] should hold values at positions {0, N/2^(i+1), 2*N/2^(i+1), ...}
    expected_step = N >> (i + 1)
    expected_positions = list(range(0, N, expected_step))
    expected_vals = (1 << (i+1)) * np.array([values[k] for k in expected_positions[:4]])
    print(f"   expected step={expected_step}, expected piece[0] coeff at 0 ≈ {expected_vals[0]:.4f}, observed={coeffs0[0]:.4f}")
