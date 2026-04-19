"""Activation profiling: hook-based recording of GELU/Softmax/LN input distributions.

Unified from profile_activations.py (class-based, Tiny-only) and
multi_model_eval.py (inline, any model). Supports any BERT variant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def build_kde_density(samples: np.ndarray, bandwidth: Optional[float] = None,
                      max_kde_samples: int = 10000):
    """Gaussian KDE density estimator (manual, no scipy dependency).

    Uses chunked computation for memory efficiency on large sample sets.
    When len(samples) > max_kde_samples, subsamples to keep KDE fast
    while preserving distribution accuracy.
    """
    # Subsample if too many points — KDE quality plateaus well before 10K
    if len(samples) > max_kde_samples:
        rng = np.random.RandomState(42)
        samples = rng.choice(samples, max_kde_samples, replace=False)

    if bandwidth is None:
        bandwidth = 1.06 * np.std(samples) * len(samples) ** (-1 / 5)
        bandwidth = max(bandwidth, 0.01)

    def density(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        n = len(samples)
        chunk_size = 5000
        for start in range(0, n, chunk_size):
            chunk = samples[start : start + chunk_size]
            diff = (x[:, None] - chunk[None, :]) / bandwidth
            result += np.exp(-0.5 * diff**2).sum(axis=1)
        result /= n * bandwidth * np.sqrt(2 * np.pi)
        return result

    return density


@torch.no_grad()
def profile_model(
    model_name: str,
    num_layers: int,
    num_samples: int = 1000,
    split: str = "train",
    model_obj: Optional[Any] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Profile activation distributions for all layers of a BERT model.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (used for tokenizer; also used to load model
        if *model_obj* is not provided).
    num_layers : int
        Number of encoder layers to profile.
    num_samples : int
        Number of SST-2 samples to run through.
    split : str
        Dataset split to use ('train' or 'validation').
    model_obj : nn.Module, optional
        An existing model to profile (e.g. fine-tuned baseline).  The encoder
        is located via ``model_obj.bert.encoder`` (classification model) or
        ``model_obj.encoder`` (plain BERT).  When provided, *model_name* is
        only used for the tokenizer.

    Returns
    -------
    dict
        {'gelu_inputs': {0: array, ...}, 'softmax_inputs': {...}, 'ln_variances': {...}}
    """
    print(f"  Profiling activations ({num_samples} samples)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use the provided model or load from model_name
    if model_obj is not None:
        model = model_obj
        owns_model = False
    else:
        model = AutoModel.from_pretrained(model_name)
        owns_model = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Locate encoder: classification models wrap in .bert
    encoder = getattr(model, "bert", model)

    dataset = load_dataset("glue", "sst2", split=split)
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    gelu_inputs: Dict[int, list] = {i: [] for i in range(num_layers)}
    softmax_inputs: Dict[int, list] = {i: [] for i in range(num_layers)}
    ln_variances: Dict[int, list] = {i: [] for i in range(num_layers)}

    hooks = []
    for layer_idx in range(num_layers):
        layer = encoder.encoder.layer[layer_idx]

        def make_gelu_hook(li):
            def hook(mod, inp, out):
                vals = out.detach().cpu().numpy().flatten()
                if len(vals) > 20_000:
                    vals = np.random.default_rng(42).choice(vals, 20_000, replace=False)
                gelu_inputs[li].append(vals)
            return hook
        hooks.append(layer.intermediate.dense.register_forward_hook(make_gelu_hook(layer_idx)))

        def make_sm_hook(li):
            def hook(mod, inp, out):
                hs = inp[0]
                Q = mod.query(hs)
                K = mod.key(hs)
                bs = Q.size(0)
                nh = mod.num_attention_heads
                hd = mod.attention_head_size
                Q = Q.view(bs, -1, nh, hd).transpose(1, 2)
                K = K.view(bs, -1, nh, hd).transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-1, -2)) / (hd**0.5)
                # Profile shifted scores (x - x.max), matching PerHeadPolynomialSoftmax.forward
                shifted = scores - scores.max(dim=-1, keepdim=True).values
                vals = shifted.detach().cpu().numpy().flatten()
                if len(vals) > 20_000:
                    vals = np.random.default_rng(42).choice(vals, 20_000, replace=False)
                softmax_inputs[li].append(vals)
            return hook
        hooks.append(layer.attention.self.register_forward_hook(make_sm_hook(layer_idx)))

        def make_ln_hook(li):
            def hook(mod, inp, out):
                x = inp[0]
                var = x.var(dim=-1, unbiased=False).detach().cpu().numpy().flatten()
                if len(var) > 5_000:
                    var = np.random.default_rng(42).choice(var, 5_000, replace=False)
                ln_variances[li].append(var)
            return hook
        hooks.append(layer.attention.output.LayerNorm.register_forward_hook(make_ln_hook(layer_idx)))
        hooks.append(layer.output.LayerNorm.register_forward_hook(make_ln_hook(layer_idx)))

    for i in range(0, len(dataset), 16):
        batch = dataset[i : i + 16]
        tokens = tokenizer(
            batch["sentence"], return_tensors="pt",
            truncation=True, padding=True, max_length=128,
        ).to(device)
        model(**tokens)

    for h in hooks:
        h.remove()

    result = {}
    for name, data_dict in [
        ("gelu_inputs", gelu_inputs),
        ("softmax_inputs", softmax_inputs),
        ("ln_variances", ln_variances),
    ]:
        result[name] = {}
        for li in range(num_layers):
            if data_dict[li]:
                result[name][li] = np.concatenate(data_dict[li])

    if owns_model:
        del model
    torch.cuda.empty_cache()
    return result


def compute_poly_coefficients(
    profile_data: Dict,
    num_layers: int,
    degree: int = 8,
) -> Dict:
    """Fit weighted minimax polynomials for each layer's operations.

    Adaptive degree scales with operation type AND layer depth.
    Intervals clamped to safe maxima to prevent polynomial blow-up.
    """
    from ..poly.approximation import weighted_minimax_approx, gelu_func, exp_func, inv_sqrt_func, gaussian_density
    from ..config import MAX_INTERVALS, FALLBACK_INTERVALS, PROFILE_KEY_MAP

    func_map = {"GELU": gelu_func, "Softmax": exp_func, "LN": inv_sqrt_func}

    poly_coeffs = {}
    for li in range(num_layers):
        for op_name in ["GELU", "Softmax", "LN"]:
            key = f"L{li}_{op_name}"
            pkey = PROFILE_KEY_MAP[op_name]

            samples = profile_data[pkey].get(li)
            if samples is None or len(samples) < 100:
                interval = FALLBACK_INTERVALS[op_name]
                density = gaussian_density(
                    center=sum(interval) / 2, std=(interval[1] - interval[0]) / 4
                )
            else:
                if op_name == "Softmax":
                    # Tighter percentiles for Softmax: shifted scores have
                    # a heavy left tail ("don't attend") that's safe to clamp.
                    p_low, p_high = np.percentile(samples, [2, 98])
                else:
                    p_low, p_high = np.percentile(samples, [0.5, 99.5])
                margin = 0.05 * (p_high - p_low)
                interval = (p_low - margin, p_high + margin)
                if op_name == "Softmax":
                    # Shifted scores are always ≤ 0; clamp upper bound
                    interval = (interval[0], min(interval[1], 0.5))
                if op_name == "LN":
                    interval = (max(0.01, interval[0]), interval[1])
                lo_max, hi_max = MAX_INTERVALS[op_name]
                interval = (max(interval[0], lo_max), min(interval[1], hi_max))
                density = build_kde_density(
                    np.random.default_rng(42).choice(
                        samples, min(10000, len(samples)), replace=False
                    )
                )

            # Depth-adaptive degree
            depth_boost = li // 4
            if op_name == "Softmax":
                deg = degree + 4 + depth_boost
            elif op_name == "GELU":
                deg = max(2, degree - 2 + depth_boost)
            else:  # LN
                deg = max(2, degree - 4 + depth_boost)

            func = func_map[op_name]
            cc, iv = weighted_minimax_approx(func, interval, deg, density)
            poly_coeffs[key] = {"cheb_coeffs": cc, "interval": iv, "degree": deg}
            print(f"    {key}: degree={deg}, interval=[{iv[0]:.2f}, {iv[1]:.2f}]")

    return poly_coeffs
