"""Load an LPAN-trained checkpoint with polynomial activations applied.

The checkpoint stored under ``results/multi_model/<model>/staged_lpan_final/best_model``
contains both the standard BERT weights and the trained polynomial
coefficients for GELU / Softmax / LayerNorm.  HuggingFace's
``from_pretrained`` does not know about the polynomial submodules, so
we have to:

1. Load the base BERT model.
2. Run model-surgery (``replace_activations``) to inject placeholder
   polynomial modules with the same shapes.
3. Re-load the checkpoint state-dict on top to restore the trained
   coefficients.

This is the same recipe used internally by ``run_staged_lpan.py`` when
resuming training; we factor it into a single helper to keep the
encryption-side validators concise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as _load_safetensors
from transformers import AutoModelForSequenceClassification

from fhe_thesis.config import MODEL_REGISTRY
from fhe_thesis.models.activations import (
    PerHeadPolynomialSoftmax,
    PolynomialGELU,
    PolynomialLayerNorm,
    PolynomialSoftmax,
)
from fhe_thesis.models.profiling import (
    compute_poly_coefficients,
    profile_model,
)
from fhe_thesis.models.replacement import replace_activations


_POLY_TYPES = (
    PolynomialGELU,
    PolynomialSoftmax,
    PerHeadPolynomialSoftmax,
    PolynomialLayerNorm,
)


def _apply_intervals_override(model: torch.nn.Module, intervals: dict) -> int:
    """Set ``module.a, module.b`` from a ``{qualified_name: [a, b]}`` dict.

    Returns the number of modules updated.  Names not present in the
    dict are left untouched.
    """
    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, _POLY_TYPES):
            continue
        if name in intervals:
            a, b = intervals[name]
            module.a = float(a)
            module.b = float(b)
            n += 1
    return n


def load_lpan_model(
    model_key: str,
    checkpoint_path: str | Path,
    *,
    num_labels: int = 2,
    device: str = "cpu",
    profile_samples: int = 200,
    degree: int = 8,
):
    """Return an ``nn.Module`` with LPAN polynomial activations restored.

    Parameters
    ----------
    model_key : {"tiny","mini","small","base"}
        Key into ``MODEL_REGISTRY``.
    checkpoint_path : str or Path
        Path to the LPAN ``best_model`` directory.
    num_labels : int
        Classifier head dimension; must match the trained checkpoint.
    device : str
        Where to materialise the model.
    profile_samples : int
        Number of samples to profile when generating placeholder
        polynomial init values.  The trained values overwrite these
        when the state-dict is loaded; only the *shapes* and
        ``interval`` metadata matter.
    degree : int
        Base polynomial degree for placeholder init.  The actual degree
        per layer is restored from the saved coefficients.
    """
    cfg = MODEL_REGISTRY[model_key]
    ckpt = Path(checkpoint_path)

    # 1. Base BERT (vanilla weights load fine; coeffs land in the
    #    "unexpected" bucket and are ignored by HuggingFace).
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt), num_labels=num_labels
    )

    # 2. Build placeholder polynomial coefficients from a fresh profile
    #    run on the base pretrained model.  We only need shape-correct
    #    placeholders; the trained values overwrite them in step 3.
    profile_data = profile_model(cfg["name"], num_layers=cfg["layers"],
                                 num_samples=profile_samples)
    poly_coeffs = compute_poly_coefficients(
        profile_data, num_layers=cfg["layers"], degree=degree
    )

    replace_activations(
        model,
        poly_coeffs,
        hidden_size=cfg["hidden"],
        learnable=True,
        replace_types=["GELU", "Softmax", "LN"],
    )

    # 3. Re-load the trained state-dict on top of the polynomial-equipped
    #    model.  `model.safetensors` is the HuggingFace default; fall
    #    back to `pytorch_model.bin` if that is missing.
    sf = ckpt / "model.safetensors"
    if sf.exists():
        state = _load_safetensors(str(sf))
    else:
        bin_path = ckpt / "pytorch_model.bin"
        state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Coefficient parameters MUST appear in the state-dict; any "missing"
    # *.coeffs is a real bug, while "missing" non-coeff entries are
    # acceptable (they belong to layers we did not replace).
    bad = [k for k in missing if "coeffs" in k]
    if bad:
        raise RuntimeError(
            f"LPAN checkpoint missing coefficient tensors: {bad[:5]}…"
        )

    # 4. Honor a frozen-intervals file if present (Stage-4 writes one).
    intervals_path = ckpt / "intervals.json"
    if intervals_path.exists():
        intervals = json.loads(intervals_path.read_text())
        n = _apply_intervals_override(model, intervals)
        print(f"  [lpan_loader] applied frozen intervals to {n} modules from {intervals_path.name}")

    return model.to(device).eval()
