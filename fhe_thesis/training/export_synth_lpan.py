"""Export a trained Stage-4 Synthesizer-LPAN checkpoint to bench JSON."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig

from fhe_thesis.models.replacement import build_poly_config_from_state_dict
from fhe_thesis.poly.chebyshev import chebyshev_to_power
from fhe_thesis.training.checkpoints import load_checkpoint_state_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Synthesizer-LPAN checkpoint to bench JSON")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory (best_model or its parent)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--intervals-json", default=None, help="Optional interval metadata JSON override")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length to export")
    return parser.parse_args()


def _normalize_checkpoint_dir(path: str) -> Path:
    ckpt_dir = Path(path)
    if (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists():
        return ckpt_dir
    candidate = ckpt_dir / "best_model"
    if (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists():
        return candidate
    raise FileNotFoundError(f"No checkpoint files found under {path}")


def _load_interval_overrides(path: Optional[str]) -> Optional[Dict[str, Sequence[float]]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): [float(v[0]), float(v[1])] for k, v in data.items()}


def _parse_layer_index(name: str) -> Optional[int]:
    marker = ".encoder.layer."
    if marker not in name:
        marker = ".transformer.layer."
        if marker not in name:
            return None
    suffix = name.split(marker, 1)[1]
    return int(suffix.split(".", 1)[0])


def _absorb_affine(power_coeffs: Sequence[float], interval: Sequence[float]) -> list[float]:
    a, b = float(interval[0]), float(interval[1])
    sscale = 2.0 / (b - a)
    sshift = -(a + b) / (b - a)
    absorbed = [0.0] * len(power_coeffs)
    for k, coeff in enumerate(power_coeffs):
        for r in range(k + 1):
            absorbed[r] += float(coeff) * (sscale ** r) * (sshift ** (k - r)) * comb(k, r)
    return absorbed


def main() -> int:
    args = _parse_args()
    checkpoint_dir = _normalize_checkpoint_dir(args.checkpoint_dir)
    interval_path = Path(args.intervals_json) if args.intervals_json else checkpoint_dir / "intervals.json"
    interval_overrides = _load_interval_overrides(str(interval_path)) if interval_path.exists() else None

    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    poly_cfg = build_poly_config_from_state_dict(state_dict, interval_overrides)
    config = AutoConfig.from_pretrained(checkpoint_dir)

    layers: Dict[int, Dict[str, object]] = {}
    for name, tensor in state_dict.items():
        if not name.endswith("pattern_logits"):
            continue
        layer_idx = _parse_layer_index(name)
        if layer_idx is None:
            continue
        logits = tensor[:, : args.seq_len, : args.seq_len].float()
        pattern = torch.softmax(logits, dim=-1).cpu().tolist()
        layers.setdefault(layer_idx, {})["attention_pattern"] = pattern

    if not layers:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} does not contain Synthesizer pattern_logits"
        )

    for layer_idx in sorted(layers):
        gelu = poly_cfg[f"L{layer_idx}_GELU"]
        ln1 = poly_cfg.get(f"L{layer_idx}_LN_attn", poly_cfg.get(f"L{layer_idx}_LN"))
        ln2 = poly_cfg.get(f"L{layer_idx}_LN_out", poly_cfg.get(f"L{layer_idx}_LN"))
        if ln1 is None or ln2 is None:
            raise KeyError(f"Missing LayerNorm coefficients for layer {layer_idx}")

        gelu_power_normalized = chebyshev_to_power(np.asarray(gelu["cheb_coeffs"], dtype=np.float64))
        ln1_power = chebyshev_to_power(np.asarray(ln1["cheb_coeffs"], dtype=np.float64))
        ln2_power = chebyshev_to_power(np.asarray(ln2["cheb_coeffs"], dtype=np.float64))

        layers[layer_idx]["gelu"] = {
            "cheb_coeffs": np.asarray(gelu["cheb_coeffs"]).tolist(),
            "interval": list(gelu["interval"]),
            "power_coeffs": _absorb_affine(gelu_power_normalized.tolist(), gelu["interval"]),
        }
        layers[layer_idx]["ln1"] = {
            "cheb_coeffs": np.asarray(ln1["cheb_coeffs"]).tolist(),
            "interval": list(ln1["interval"]),
            "power_coeffs": ln1_power.tolist(),
        }
        layers[layer_idx]["ln2"] = {
            "cheb_coeffs": np.asarray(ln2["cheb_coeffs"]).tolist(),
            "interval": list(ln2["interval"]),
            "power_coeffs": ln2_power.tolist(),
        }

    payload = {
        "metadata": {
            "checkpoint_dir": str(checkpoint_dir),
            "seq_len": args.seq_len,
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "intermediate_size": int(config.intermediate_size),
            "model_type": str(config.model_type),
        },
        "layers": [
            {"layer_idx": idx, **layers[idx]}
            for idx in sorted(layers)
        ],
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else checkpoint_dir.parent / "bench_checkpoint.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())