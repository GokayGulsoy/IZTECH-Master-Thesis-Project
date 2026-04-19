"""Model-agnostic LPAN-FHE protocol orchestration.

Builds an end-to-end encrypted forward pass for any BERT variant in
``fhe_thesis.config.MODEL_REGISTRY``. All shapes (num_layers, hidden,
heads) are read from the model's HuggingFace config, so the same code
path runs on Tiny / Mini / Small / Base.

Public API
----------
* :func:`encrypt_ffn_block`        — single FFN + LN block, one layer
* :func:`encrypt_attention_block`  — one MHA block, one layer
* :func:`encrypt_layer`            — full transformer layer (attn + FFN)
* :func:`encrypt_inference`        — full encoder + classifier head

Each routine takes a plaintext numpy input, runs the encrypted
pipeline under the supplied :class:`CKKSBackend`, decrypts at the end,
and returns the numpy result plus a latency dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from fhe_thesis.config import MODEL_REGISTRY

from .backend import CKKSBackend
from .coefficients import PolyCoeffs, load_coefficients
from .ops import (
    enc_gelu_poly,
    enc_linear,
    enc_ln_poly,
    enc_self_attention,
)
from .packing import TokenPackedTensor


# ──────────────────────────────────────────────────────────────────────
# Weight bundle pulled from a HuggingFace BertModel
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LayerWeights:
    """Plaintext weights for one BERT encoder layer."""

    Wq: np.ndarray
    bq: np.ndarray
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray  # post-attn LN
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray  # post-FFN LN


@dataclass
class ModelWeights:
    """All encoder layers + classifier head for a BERT variant."""

    model_key: str
    num_layers: int
    hidden: int
    num_heads: int
    layers: List[LayerWeights] = field(default_factory=list)
    pooler_W: np.ndarray | None = None  # CLS pooler
    pooler_b: np.ndarray | None = None
    cls_W: np.ndarray | None = None  # classifier head
    cls_b: np.ndarray | None = None


def load_model_weights(
    model_key: str,
    *,
    checkpoint_path: str | None = None,
    num_labels: int = 2,
) -> ModelWeights:
    """Pull plaintext weights from a HuggingFace BertForSequenceClassification.

    If ``checkpoint_path`` is None we use the original pretrained model
    from ``MODEL_REGISTRY``. Otherwise we load the LPAN-trained
    checkpoint, which has the same parameter shapes.
    """
    from transformers import AutoModelForSequenceClassification  # heavy import

    cfg = MODEL_REGISTRY[model_key]
    src = checkpoint_path or cfg["name"]
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=num_labels
    )
    sd = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}

    layers: List[LayerWeights] = []
    for i in range(cfg["layers"]):
        p = f"bert.encoder.layer.{i}"
        layers.append(
            LayerWeights(
                Wq=sd[f"{p}.attention.self.query.weight"],
                bq=sd[f"{p}.attention.self.query.bias"],
                Wk=sd[f"{p}.attention.self.key.weight"],
                bk=sd[f"{p}.attention.self.key.bias"],
                Wv=sd[f"{p}.attention.self.value.weight"],
                bv=sd[f"{p}.attention.self.value.bias"],
                Wo=sd[f"{p}.attention.output.dense.weight"],
                bo=sd[f"{p}.attention.output.dense.bias"],
                W1=sd[f"{p}.intermediate.dense.weight"],
                b1=sd[f"{p}.intermediate.dense.bias"],
                W2=sd[f"{p}.output.dense.weight"],
                b2=sd[f"{p}.output.dense.bias"],
                ln1_gamma=sd[f"{p}.attention.output.LayerNorm.weight"],
                ln1_beta=sd[f"{p}.attention.output.LayerNorm.bias"],
                ln2_gamma=sd[f"{p}.output.LayerNorm.weight"],
                ln2_beta=sd[f"{p}.output.LayerNorm.bias"],
            )
        )

    return ModelWeights(
        model_key=model_key,
        num_layers=cfg["layers"],
        hidden=cfg["hidden"],
        num_heads=cfg["heads"],
        layers=layers,
        pooler_W=sd.get("bert.pooler.dense.weight"),
        pooler_b=sd.get("bert.pooler.dense.bias"),
        cls_W=sd.get("classifier.weight"),
        cls_b=sd.get("classifier.bias"),
    )


# ──────────────────────────────────────────────────────────────────────
# Encrypted block primitives (model-agnostic)
# ──────────────────────────────────────────────────────────────────────


def encrypt_ffn_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run FFN + post-LN under FHE.

    Pipeline: ``W₁ → GELU-poly → W₂ → residual → LN-poly``.
    """
    timings: Dict[str, float] = {}

    t = time.time()
    h = enc_linear(backend, x, layer.W1, layer.b1)
    timings["W1"] = time.time() - t

    t = time.time()
    g = coeffs["GELU"]
    h = enc_gelu_poly(backend, h, g.power_coeffs, g.interval)
    timings["GELU"] = time.time() - t

    t = time.time()
    h = enc_linear(backend, h, layer.W2, layer.b2)
    timings["W2"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln2_gamma,
        beta=layer.ln2_beta,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_attention_block(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run MHA + post-LN under FHE.

    Pipeline: ``MHA → residual → LN-poly``.
    """
    timings: Dict[str, float] = {}

    t = time.time()
    sm = coeffs["Softmax"]
    h = enc_self_attention(
        backend,
        x,
        layer.Wq,
        layer.bq,
        layer.Wk,
        layer.bk,
        layer.Wv,
        layer.bv,
        layer.Wo,
        layer.bo,
        softmax_power_coeffs=sm.power_coeffs,
        softmax_interval=sm.interval,
        num_heads=num_heads,
    )
    timings["MHA"] = time.time() - t

    t = time.time()
    res = TokenPackedTensor.from_ciphertexts(
        [backend.add(h.cts[i], x.cts[i]) for i in range(x.seq_len)],
        hidden_dim=x.hidden_dim,
    )
    timings["residual"] = time.time() - t

    t = time.time()
    ln = coeffs["LN"]
    out = enc_ln_poly(
        backend,
        res,
        ln.power_coeffs,
        ln.interval,
        gamma=layer.ln1_gamma,
        beta=layer.ln1_beta,
    )
    timings["LN"] = time.time() - t
    return out, timings


def encrypt_layer(
    backend: CKKSBackend,
    x: TokenPackedTensor,
    layer: LayerWeights,
    coeffs: Dict[str, PolyCoeffs],
    num_heads: int,
) -> Tuple[TokenPackedTensor, Dict[str, float]]:
    """Run one full BERT encoder layer under FHE."""
    h, t_attn = encrypt_attention_block(backend, x, layer, coeffs, num_heads)
    h, t_ffn = encrypt_ffn_block(backend, h, layer, coeffs)
    timings = {f"attn.{k}": v for k, v in t_attn.items()}
    timings.update({f"ffn.{k}": v for k, v in t_ffn.items()})
    return h, timings


def encrypt_inference(
    backend: CKKSBackend,
    x_plain: np.ndarray,
    weights: ModelWeights,
    coeffs: Dict[int, Dict[str, PolyCoeffs]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Full encoder + (optional) classifier head under FHE.

    The classifier head is a plaintext Linear; we apply it with one
    final ``enc_linear`` if ``weights.cls_W`` is set. Returns the
    decrypted output and a flat latency dict.
    """
    timings: Dict[str, float] = {}

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings["encrypt"] = time.time() - t

    h = ct_x
    for i, layer in enumerate(weights.layers):
        h, layer_t = encrypt_layer(backend, h, layer, coeffs[i], weights.num_heads)
        for k, v in layer_t.items():
            timings[f"L{i}.{k}"] = v

    # Pooler + classifier on the [CLS] token only.
    if weights.cls_W is not None:
        t = time.time()
        cls = TokenPackedTensor.from_ciphertexts([h.cts[0]], hidden_dim=h.hidden_dim)
        if weights.pooler_W is not None:
            cls = enc_linear(backend, cls, weights.pooler_W, weights.pooler_b)
            # tanh is non-poly; the LPAN pipeline trains with the pooler
            # frozen so we approximate it as identity here. (See thesis
            # §3 — the [CLS] head is folded into the classifier in
            # downstream LPAN runs.)
        out_ct = enc_linear(backend, cls, weights.cls_W, weights.cls_b)
        timings["classifier"] = time.time() - t
    else:
        out_ct = h

    t = time.time()
    out = out_ct.decrypt(backend)
    timings["decrypt"] = time.time() - t

    timings["total"] = sum(timings.values())
    return out, timings


# ──────────────────────────────────────────────────────────────────────
# Convenience: one-call helper for a model + phase
# ──────────────────────────────────────────────────────────────────────


def run_phase(
    phase: str,
    model_key: str,
    backend: CKKSBackend,
    x_plain: np.ndarray,
    *,
    layer_idx: int = 0,
    checkpoint_path: str | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Dispatch helper used by ``experiments/run_protocol.py``.

    ``phase ∈ {"ffn", "attention", "layer", "model"}``.
    """
    weights = load_model_weights(model_key, checkpoint_path=checkpoint_path)
    coeffs = load_coefficients(model_key)

    if phase == "model":
        return encrypt_inference(backend, x_plain, weights, coeffs)

    layer = weights.layers[layer_idx]
    layer_coeffs = coeffs[layer_idx]
    timings_total: Dict[str, float] = {}

    t = time.time()
    ct_x = TokenPackedTensor.encrypt(backend, x_plain)
    timings_total["encrypt"] = time.time() - t

    if phase == "ffn":
        out_ct, t_block = encrypt_ffn_block(backend, ct_x, layer, layer_coeffs)
    elif phase == "attention":
        out_ct, t_block = encrypt_attention_block(
            backend, ct_x, layer, layer_coeffs, weights.num_heads
        )
    elif phase == "layer":
        out_ct, t_block = encrypt_layer(
            backend, ct_x, layer, layer_coeffs, weights.num_heads
        )
    else:
        raise ValueError(f"unknown phase {phase!r}")
    timings_total.update(t_block)

    t = time.time()
    out = out_ct.decrypt(backend)
    timings_total["decrypt"] = time.time() - t
    timings_total["total"] = sum(timings_total.values())
    return out, timings_total
