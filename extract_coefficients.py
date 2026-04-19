#!/usr/bin/env python3
"""Extract learned polynomial coefficients from staged-LPAN models.

Loads the final (Stage 3) checkpoint for each BERT variant and writes
per-model JSON files to results/coefficients/.

Usage:
    python extract_coefficients.py                 # all models
    python extract_coefficients.py --model base    # single model
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


MODELS = {
    "tiny":  "results/multi_model/tiny/staged_lpan_final/best_model",
    "mini":  "results/multi_model/mini/staged_lpan_final/best_model",
    "small": "results/multi_model/small/staged_lpan_final/best_model",
    "base":  "results/multi_model/base/staged_lpan_final/best_model",
}

OUT_DIR = Path("results/coefficients")


def extract(model_dir: str) -> dict:
    """Return {layer_key: {degree, coeffs, activation_type}} from a checkpoint."""
    sf = Path(model_dir) / "model.safetensors"
    if sf.exists():
        state = load_file(str(sf))
    else:
        import torch
        state = torch.load(
            Path(model_dir) / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
        )

    result = {}
    for name in sorted(state.keys()):
        if "coeffs" not in name:
            continue
        vals = state[name].tolist()

        # Determine activation type from the parameter path
        if "intermediate_act_fn" in name:
            act_type = "gelu"
        elif "LayerNorm" in name:
            act_type = "layernorm"
        elif "poly_softmax" in name:
            act_type = "softmax"
        else:
            act_type = "unknown"

        # Extract layer index
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break

        result[name] = {
            "activation_type": act_type,
            "layer": layer_idx,
            "degree": len(vals) - 1,
            "coefficients": [round(v, 8) for v in vals],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract polynomial coefficients")
    parser.add_argument(
        "--model",
        nargs="*",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which model(s) to extract (default: all)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.model:
        model_dir = MODELS[name]
        if not Path(model_dir).exists():
            print(f"  SKIP {name}: {model_dir} not found")
            continue

        coeffs = extract(model_dir)
        out_path = OUT_DIR / f"bert_{name}_coeffs.json"
        with open(out_path, "w") as f:
            json.dump(coeffs, f, indent=2)

        n_gelu = sum(1 for v in coeffs.values() if v["activation_type"] == "gelu")
        n_ln   = sum(1 for v in coeffs.values() if v["activation_type"] == "layernorm")
        n_soft = sum(1 for v in coeffs.values() if v["activation_type"] == "softmax")
        degrees = sorted(set(v["degree"] for v in coeffs.values()))

        print(f"  BERT-{name.capitalize()}: {len(coeffs)} polynomials "
              f"(GELU={n_gelu}, LN={n_ln}, Softmax={n_soft}), "
              f"degrees={degrees} → {out_path}")


if __name__ == "__main__":
    main()
