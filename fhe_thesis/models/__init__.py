"""Polynomial activation modules, model surgery, and activation profiling."""
from .activations import (
	PolynomialGELU,
	PolynomialLayerNorm,
	PolynomialSoftmax,
	SynthesizerAttention,
)
from .replacement import replace_activations, replace_attention_with_synthesizer
from .profiling import profile_model, build_kde_density, compute_poly_coefficients
