#!/usr/bin/env python3
"""
Experiment 01: Polynomial Approximation Comparison (Contribution 1)
====================================================================
Compares Taylor, Chebyshev, Least-Squares, and Weighted Minimax (ours)
for GELU, exp, and 1/sqrt(x).

Outputs → results/poly_approx/
"""
from fhe_thesis.config import POLY_APPROX_DIR, ensure_dirs
from fhe_thesis.poly.approximation import (
    gelu_func, exp_func, inv_sqrt_func,
    gaussian_density, shifted_exp_density, variance_density,
    compare_approximations, print_results_table,
)
import json


def main():
    ensure_dirs()
    print("=" * 70)
    print("  Polynomial Approximation Toolkit for FHE-Compatible Activations")
    print("=" * 70)

    degrees = [2, 3, 4, 5, 6, 7, 8]

    print("\n[1/3] Computing GELU approximations...")
    gelu_results = compare_approximations(
        func=gelu_func, func_name="GELU", interval=(-5.0, 5.0),
        degrees=degrees, density_func=gaussian_density(0.0, 1.5),
        taylor_center=0.0, output_dir=POLY_APPROX_DIR,
    )
    print_results_table(gelu_results, "GELU [-5, 5]")

    print("\n[2/3] Computing exp(x) approximations...")
    exp_results = compare_approximations(
        func=exp_func, func_name="Exponential", interval=(-8.0, 0.0),
        degrees=degrees, density_func=shifted_exp_density(-4.0, 2.0),
        taylor_center=-4.0, output_dir=POLY_APPROX_DIR,
    )
    print_results_table(exp_results, "exp(x) [-8, 0]")

    print("\n[3/3] Computing 1/sqrt(x) approximations...")
    invsqrt_results = compare_approximations(
        func=inv_sqrt_func, func_name="InvSqrt", interval=(0.1, 4.0),
        degrees=degrees, density_func=variance_density(1.0, 0.5),
        taylor_center=1.0, output_dir=POLY_APPROX_DIR,
    )
    print_results_table(invsqrt_results, "1/√x [0.1, 4.0]")

    all_results = {"GELU": gelu_results, "Exponential": exp_results, "InvSqrt": invsqrt_results}
    with open(POLY_APPROX_DIR / "numerical_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  All results saved to: {POLY_APPROX_DIR}/")


if __name__ == "__main__":
    main()
