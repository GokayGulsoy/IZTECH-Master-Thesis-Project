#!/usr/bin/env python3
"""
Experiment 09: Polynomial Evaluation Strategies (BSGS) for CKKS
=================================================================
Compares Horner, Balanced Tree, and Paterson-Stockmeyer polynomial
evaluation methods under CKKS encryption.

Outputs → results/bsgs_eval/
"""
from __future__ import annotations

import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tenseal as ts

from fhe_thesis.config import BSGS_EVAL_DIR, ensure_dirs
from fhe_thesis.encryption.context import make_context

OUT = BSGS_EVAL_DIR


# ── Evaluation methods ───────────────────────────────────────────────────────

def horner_encrypted(ctx, coeffs, x_plain):
    n = len(coeffs) - 1
    x_enc = ts.ckks_vector(ctx, x_plain)
    t0 = time.perf_counter()
    result = ts.ckks_vector(ctx, [coeffs[n]] * len(x_plain))
    ct_mults = 0
    for i in range(n - 1, -1, -1):
        result = result * x_enc; ct_mults += 1
        result = result + [coeffs[i]] * len(x_plain)
    t1 = time.perf_counter()
    return result.decrypt(), (t1 - t0) * 1000, n, ct_mults


def balanced_tree_encrypted(ctx, coeffs, x_plain):
    n = len(coeffs) - 1
    x_enc = ts.ckks_vector(ctx, x_plain)
    t0 = time.perf_counter()
    ct_mults = 0
    powers = {1: x_enc}; depth_at = {1: 0}
    p = 1
    while p * 2 <= n:
        p2 = p * 2
        powers[p2] = powers[p] * powers[p]; depth_at[p2] = depth_at[p] + 1; ct_mults += 1
        p = p2
    for k in range(2, n + 1):
        if k not in powers:
            a = 1
            while a * 2 <= k: a *= 2
            b_ = k - a
            if b_ not in powers:
                for j in range(2, b_ + 1):
                    if j not in powers:
                        powers[j] = powers[j-1] * powers[1]
                        depth_at[j] = max(depth_at.get(j-1, 0), depth_at[1]) + 1; ct_mults += 1
            powers[k] = powers[a] * powers[b_]
            depth_at[k] = max(depth_at[a], depth_at[b_]) + 1; ct_mults += 1
    result = powers[1] * coeffs[1]
    for i in range(2, n + 1):
        result = result + powers[i] * coeffs[i]
    result = result + coeffs[0]
    t1 = time.perf_counter()
    return result.decrypt(), (t1 - t0) * 1000, max(depth_at.values()), ct_mults


def paterson_stockmeyer_encrypted(ctx, coeffs, x_plain):
    n = len(coeffs) - 1
    k = math.isqrt(n) + (1 if math.isqrt(n) ** 2 < n else 0)
    if k < 2: k = 2
    x_enc = ts.ckks_vector(ctx, x_plain)
    t0 = time.perf_counter()
    ct_mults = 0

    baby = {1: x_enc}; baby_depth = {1: 0}
    p = 1
    while p * 2 <= k:
        p2 = p * 2
        baby[p2] = baby[p] * baby[p]; baby_depth[p2] = baby_depth[p] + 1; ct_mults += 1
        p = p2
    for j in range(2, k + 1):
        if j not in baby:
            a = 1
            while a * 2 <= j: a *= 2
            b_ = j - a
            if b_ not in baby:
                baby[b_] = baby[b_-1] * baby[1]; baby_depth[b_] = max(baby_depth.get(b_-1,0), baby_depth[1]) + 1; ct_mults += 1
            baby[j] = baby[a] * baby[b_]; baby_depth[j] = max(baby_depth[a], baby_depth[b_]) + 1; ct_mults += 1

    num_groups = (n // k) + 1
    q_results = []
    for g in range(num_groups):
        max_i = min(k - 1, n - g * k)
        acc = None
        for i in range(1, max_i + 1):
            idx = g * k + i
            if idx <= n:
                term = baby[i] * coeffs[idx]
                acc = term if acc is None else acc + term
        c0 = coeffs[g * k] if g * k <= n else 0.0
        if acc is None:
            acc = baby[1] * 0.0 + c0
        else:
            acc = acc + c0
        q_results.append(acc)

    if num_groups > 1:
        giant = {1: baby[k]}; giant_depth = {1: baby_depth[k]}
        for g in range(2, num_groups):
            giant[g] = giant[g-1] * giant[1]; giant_depth[g] = max(giant_depth[g-1], giant_depth[1]) + 1; ct_mults += 1

    result = q_results[0]
    for g in range(1, num_groups):
        result = result + q_results[g] * giant[g]; ct_mults += 1
    t1 = time.perf_counter()

    baby_max = max(baby_depth.values())
    giant_max = max(giant_depth.values()) if num_groups > 1 else 0
    total_depth = baby_max + giant_max + (1 if num_groups > 1 else 0)
    return result.decrypt(), (t1 - t0) * 1000, total_depth, ct_mults


def benchmark_degree(degree, x_vals):
    print(f"\n  Degree {degree}:")
    np.random.seed(42 + degree)
    coeffs = np.random.randn(degree + 1).tolist()
    mx = max(abs(c) for c in coeffs)
    coeffs = [c / mx for c in coeffs]

    plain_result = np.polyval(coeffs[::-1], x_vals)
    x_list = x_vals.tolist()
    results = {"degree": degree, "methods": {}}

    methods = {"Balanced Tree": balanced_tree_encrypted,
               "Paterson-Stockmeyer": paterson_stockmeyer_encrypted}
    if degree <= 8:
        methods["Horner"] = horner_encrypted

    for name, func in methods.items():
        try:
            if name == "Horner": req_depth = degree
            elif name == "Balanced Tree": req_depth = math.ceil(math.log2(max(degree, 2))) + 1
            else:
                ks = math.isqrt(degree) + (1 if math.isqrt(degree)**2 < degree else 0)
                req_depth = math.ceil(math.log2(max(ks, 2))) + math.ceil(degree / ks) + 2
            ctx = make_context(req_depth)
            dec, time_ms, depth, n_mults = func(ctx, coeffs, x_list)
            err = np.abs(np.array(dec) - plain_result)
            results["methods"][name] = {
                "time_ms": round(time_ms, 2), "depth": depth,
                "ct_ct_mults": n_mults, "max_error": float(np.max(err)),
            }
            print(f"    {name:25s}: depth={depth}, mults={n_mults:3d}, "
                  f"time={time_ms:8.1f}ms, err={np.max(err):.2e}")
        except Exception as e:
            results["methods"][name] = {"status": "failed", "error": str(e)}
            print(f"    {name:25s}: FAILED — {e}")

    return results


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Polynomial Evaluation Strategies (BSGS) for CKKS")
    print("=" * 70)

    x_vals = np.linspace(-1, 1, 64)
    degrees = [2, 4, 6, 8, 12, 16]
    all_results = {}

    for deg in degrees:
        all_results[deg] = benchmark_degree(deg, x_vals)

    # Theoretical comparison
    theory = {}
    for n in [2, 4, 6, 8, 12, 16, 24, 32]:
        ks = math.isqrt(n) + (1 if math.isqrt(n)**2 < n else 0)
        theory[n] = {
            "Horner": {"depth": n, "mults": n},
            "Balanced Tree": {"depth": math.ceil(math.log2(max(n, 2))),
                              "mults": n + math.ceil(math.log2(max(n, 2)))},
            "P-S (BSGS)": {"depth": math.ceil(math.log2(max(ks, 2))) +
                                    math.ceil(n / ks),
                           "mults": ks + math.ceil(n / ks)},
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    degs_t = sorted(theory.keys())
    for method, color, marker in [("Horner", "tab:red", "o"),
                                   ("Balanced Tree", "tab:blue", "s"),
                                   ("P-S (BSGS)", "tab:green", "D")]:
        ax1.plot(degs_t, [theory[d][method]["depth"] for d in degs_t],
                 f"{marker}-", color=color, label=method, linewidth=2)
        ax2.plot(degs_t, [theory[d][method]["mults"] for d in degs_t],
                 f"{marker}-", color=color, label=method, linewidth=2)
    ax1.set(xlabel="Polynomial Degree", ylabel="Multiplicative Depth",
            title="Depth Complexity"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Polynomial Degree", ylabel="CT-CT Multiplications",
            title="Multiplication Count"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "bsgs_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved bsgs_comparison.png")

    with open(OUT / "bsgs_results.json", "w") as f:
        json.dump({"empirical": {str(k): v for k, v in all_results.items()},
                   "theoretical": {str(k): v for k, v in theory.items()}},
                  f, indent=2, default=str)

    print(f"  All results saved to: {OUT}/")


if __name__ == "__main__":
    main()
