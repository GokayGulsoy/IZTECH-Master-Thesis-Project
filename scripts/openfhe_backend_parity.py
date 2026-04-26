"""Parity test: OpenFHE backend vs. TenSEAL backend.

Runs the same low-level ops on both backends and verifies numerical
agreement with the plaintext reference.

Pass criterion: max abs error < 1e-2 on each operation.
"""
from __future__ import annotations

import math
import time

import numpy as np


def main() -> None:
    print("=== Backend parity: OpenFHE vs TenSEAL ===\n")

    from fhe_thesis.encryption.backend import TenSEALBackend
    from fhe_thesis.encryption.openfhe_backend import OpenFHEBackend

    # --- backends ---------------------------------------------------
    print("[setup] TenSEAL backend ...")
    ts_back = TenSEALBackend(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
        global_scale_bits=40,
    )
    print("[setup] OpenFHE backend (no bootstrap, fast smoke) ...")
    import openfhe as ofhe

    of_back = OpenFHEBackend(
        multiplicative_depth=8,
        ring_dim=1 << 13,
        scaling_mod_size=40,
        first_mod_size=60,
        enable_bootstrap=False,
        num_slots=8,
        security_level=ofhe.SecurityLevel.HEStd_NotSet,
    )
    print("  setup done.\n")

    # --- inputs -----------------------------------------------------
    n = 8
    x = np.array([0.5, -0.3, 0.7, 0.1, -0.6, 0.25, -0.15, 0.4])
    y = np.array([0.2, 0.8, -0.5, 0.6, -0.1, 0.45, 0.05, -0.7])
    weight = np.array(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.05, 0.15, 0.25],
            [-0.3, 0.4, 0.1, 0.2, -0.5, 0.05, -0.1, 0.35],
            [0.6, -0.1, 0.0, 0.3, -0.2, 0.45, -0.25, 0.1],
            [0.0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
        ]
    )
    bias = np.array([0.01, -0.02, 0.03, -0.04])
    # Power-basis polynomial coefficients (deg 3): p(x) = 0.5 + 0.3 x + 0.1 x² - 0.05 x³
    poly_coeffs = [0.5, 0.3, 0.1, -0.05]

    # --- helpers ----------------------------------------------------
    def cmp(name, ts_out, of_out, ref, k=n):
        ts_v = np.array(ts_out[:k])
        of_v = np.array(of_out[:k])
        ref_v = np.array(ref[:k])
        err_ts = np.max(np.abs(ts_v - ref_v))
        err_of = np.max(np.abs(of_v - ref_v))
        ok_ts = "OK" if err_ts < 1e-2 else "FAIL"
        ok_of = "OK" if err_of < 1e-2 else "FAIL"
        print(
            f"  {name:<14s}  ts_err={err_ts:.2e} {ok_ts}    "
            f"of_err={err_of:.2e} {ok_of}"
        )
        return err_of < 1e-2

    # --- 1. encrypt / decrypt round-trip ---------------------------
    ct_ts = ts_back.encrypt(x.tolist())
    ct_of = of_back.encrypt(x.tolist())
    cmp("encrypt", ts_back.decrypt(ct_ts), of_back.decrypt(ct_of), x)

    # --- 2. add / sub ----------------------------------------------
    ct2_ts = ts_back.encrypt(y.tolist())
    ct2_of = of_back.encrypt(y.tolist())
    cmp(
        "add",
        ts_back.decrypt(ts_back.add(ct_ts, ct2_ts)),
        of_back.decrypt(of_back.add(ct_of, ct2_of)),
        x + y,
    )
    cmp(
        "sub",
        ts_back.decrypt(ts_back.sub(ct_ts, ct2_ts)),
        of_back.decrypt(of_back.sub(ct_of, ct2_of)),
        x - y,
    )

    # --- 3. add_plain / mul_plain ----------------------------------
    cmp(
        "add_plain",
        ts_back.decrypt(ts_back.add_plain(ct_ts, y.tolist())),
        of_back.decrypt(of_back.add_plain(ct_of, y.tolist())),
        x + y,
    )
    cmp(
        "mul_plain",
        ts_back.decrypt(ts_back.mul_plain(ct_ts, y.tolist())),
        of_back.decrypt(of_back.mul_plain(ct_of, y.tolist())),
        x * y,
    )

    # --- 4. ct·ct mul ---------------------------------------------
    cmp(
        "mul",
        ts_back.decrypt(ts_back.mul(ct_ts, ct2_ts)),
        of_back.decrypt(of_back.mul(ct_of, ct2_of)),
        x * y,
    )

    # --- 5. polyval -----------------------------------------------
    p_ref = sum(c * x**i for i, c in enumerate(poly_coeffs))
    cmp(
        "polyval",
        ts_back.decrypt(ts_back.polyval(ct_ts, poly_coeffs)),
        of_back.decrypt(of_back.polyval(ct_of, poly_coeffs)),
        p_ref,
    )

    # --- 6. matmul_plain -------------------------------------------
    ref = weight @ x + bias
    cmp(
        "matmul_plain",
        ts_back.decrypt(ts_back.matmul_plain(ct_ts, weight.tolist(), bias.tolist())),
        of_back.decrypt(of_back.matmul_plain(ct_of, weight.tolist(), bias.tolist())),
        ref,
        k=4,
    )

    # --- 7. dot ----------------------------------------------------
    cmp(
        "dot (slot 0)",
        ts_back.decrypt(ts_back.dot(ct_ts, ct2_ts))[:1],
        of_back.decrypt(of_back.dot(ct_of, ct2_of))[:1],
        [float(x @ y)],
        k=1,
    )

    # --- 8. sum_slots ---------------------------------------------
    cmp(
        "sum_slots",
        ts_back.decrypt(ts_back.sum_slots(ct_ts))[:1],
        of_back.decrypt(of_back.sum_slots(ct_of))[:1],
        [float(x.sum())],
        k=1,
    )

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
