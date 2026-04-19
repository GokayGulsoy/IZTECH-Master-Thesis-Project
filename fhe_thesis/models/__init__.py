"""Polynomial activation modules, model surgery, and activation profiling."""
from .activations import PolynomialGELU, PolynomialSoftmax, PolynomialLayerNorm
from .replacement import replace_activations
from .profiling import profile_model, build_kde_density, compute_poly_coefficients
