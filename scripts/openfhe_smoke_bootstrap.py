"""Smoke test: OpenFHE CKKS bootstrap on this system.

Based on the official openfhe-python CKKS bootstrap example.
"""
from __future__ import annotations

import time

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    SecretKeyDist,
    SecurityLevel,
)


def main() -> None:
    print("=== OpenFHE CKKS bootstrap smoke test ===")

    level_budget = [3, 3]
    levels_after_boot = 10
    boot_depth = 9 + sum(level_budget)
    depth = levels_after_boot + boot_depth

    parameters = CCParamsCKKSRNS()
    parameters.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    parameters.SetRingDim(1 << 12)
    parameters.SetScalingModSize(59)
    parameters.SetFirstModSize(60)
    parameters.SetMultiplicativeDepth(depth)
    num_slots = 8
    parameters.SetBatchSize(num_slots)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(PKESchemeFeature.FHE)

    print(f"  ring dim          = {cc.GetRingDimension()}")
    print(f"  multiplicative D  = {depth}")

    cc.EvalBootstrapSetup(level_budget, [0, 0], num_slots)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, num_slots)
    print("  bootstrap keys    : OK")

    x = [0.25, -0.5, 0.1, 0.7, -0.3, 0.55, -0.05, 0.4]
    pt = cc.MakeCKKSPackedPlaintext(x, 1, depth - 1, None, num_slots)
    ct = cc.Encrypt(keys.publicKey, pt)
    print(f"  initial level     = {ct.GetLevel()}")

    print("  running bootstrap ...")
    t0 = time.perf_counter()
    ct_refreshed = cc.EvalBootstrap(ct)
    boot_time = time.perf_counter() - t0
    print(f"  bootstrap time    = {boot_time:.2f} s")
    print(f"  level after boot  = {ct_refreshed.GetLevel()}")

    pt_out = cc.Decrypt(ct_refreshed, keys.secretKey)
    pt_out.SetLength(len(x))
    decoded = pt_out.GetRealPackedValue()
    print("  decoded =", [f"{v:+.4f}" for v in decoded])
    print("  expected=", [f"{v:+.4f}" for v in x])
    err = max(abs(a - b) for a, b in zip(decoded, x))
    print(f"  max abs error     = {err:.5f}")

    assert err < 5e-2, f"Bootstrap error too large: {err}"
    print("=== PASSED ===")


if __name__ == "__main__":
    main()
