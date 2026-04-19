"""Polynomial approximation and evaluation modules."""
from .chebyshev import cheb_eval_torch, chebyshev_to_power
from .approximation import (
    gelu_func, exp_func, inv_sqrt_func,
    weighted_minimax_approx, chebyshev_approx, eval_chebyshev,
    taylor_approx, least_squares_approx,
    gaussian_density, shifted_exp_density, variance_density,
    compute_errors, multiplicative_depth,
    compare_approximations, print_results_table,
)
