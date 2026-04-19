"""CKKS context creation and encrypted polynomial operations.

Unified from encrypted_inference.py and bsgs_poly_eval.py.
"""

from __future__ import annotations

from typing import List, Optional

import tenseal as ts


def create_ckks_context(
    poly_modulus_degree: int = 16384,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
    global_scale_bits: int = 40,
) -> ts.Context:
    """Create a TenSEAL CKKS context with appropriate parameters.

    Parameters
    ----------
    poly_modulus_degree : int
        Ring dimension N (default 16384).
    coeff_mod_bit_sizes : list[int] or None
        Coefficient modulus chain. Default: 7 levels.
    global_scale_bits : int
        Bit size of Δ = 2^global_scale_bits.
    """
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 40, 60]

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    ctx.global_scale = 2 ** global_scale_bits
    ctx.generate_galois_keys()
    return ctx


def make_context(mult_depth: int) -> ts.Context:
    """Create CKKS context with enough levels for the given multiplicative depth.

    Auto-computes poly_modulus_degree from total bit budget.
    """
    coeff_bits = [60] + [40] * mult_depth + [60]
    total_bits = sum(coeff_bits)

    if total_bits <= 218:
        poly_mod = 8192
    elif total_bits <= 438:
        poly_mod = 16384
    else:
        poly_mod = 32768

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coeff_bits,
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    return ctx
