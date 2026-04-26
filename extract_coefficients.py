#!/usr/bin/env python3
"""Extract learned polynomial coefficients from staged-LPAN models.

Loads the final (Stage 3) checkpoint for each (model, task) pair and
writes per-combo JSON files to ``results/coefficients/``.

Output filename convention
--------------------------
* All tasks use a per-task subdirectory:
  ``results/coefficients/{task}/bert_{model}_coeffs.json``.

Usage:
    python extract_coefficients.py                            # all models, sst2
    python extract_coefficients.py --model base               # single model
    python extract_coefficients.py --task mrpc                # all models on MRPC
    python extract_coefficients.py --task all --model all     # full 4×4 sweep
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


MODEL_KEYS = ["tiny", "mini", "small", "base"]
TASK_KEYS = ["sst2", "mrpc", "qnli", "qqp"]

OUT_DIR = Path("results/coefficients")


def _ckpt_path(model: str, task: str, prefer_stage4: bool = True) -> Path:
    """Resolve the LPAN checkpoint for (model, task).

    All tasks use the unified ``{task}/{model}/`` layout.
    Prefers ``stage4_range_aware/best_model`` when ``prefer_stage4=True``
    and that checkpoint exists (FHE-deployable, inference-aware intervals);
    falls back to ``staged_lpan_final/best_model`` otherwise.
    """
    base = Path("results/multi_model") / task / model
    if prefer_stage4:
        s4 = base / "stage4_range_aware" / "best_model"
        if s4.exists():
            return s4
    return base / "staged_lpan_final" / "best_model"


def _out_path(model: str, task: str) -> Path:
    """Mirror ``_ckpt_path``'s task-aware layout for the JSON output."""
    return OUT_DIR / task / f"bert_{model}_coeffs.json"


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
        tensor = state[name]

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

        # Handle per-head softmax coefficients (2D: num_heads × degree+1)
        if tensor.dim() == 2:
            vals = [[round(v, 8) for v in row] for row in tensor.tolist()]
            degree = tensor.shape[1] - 1
            num_heads = tensor.shape[0]
        else:
            vals = [round(v, 8) for v in tensor.tolist()]
            degree = len(vals) - 1
            num_heads = None

        entry = {
            "activation_type": act_type,
            "layer": layer_idx,
            "degree": degree,
            "coefficients": vals,
        }
        if num_heads is not None:
            entry["num_heads"] = num_heads
        result[name] = entry

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract polynomial coefficients")
    parser.add_argument(
        "--model",
        nargs="*",
        default=MODEL_KEYS,
        help="Which model(s) to extract — keys or 'all' (default: all)",
    )
    parser.add_argument(
        "--task",
        nargs="*",
        default=["sst2"],
        help="Which GLUE task(s) — keys or 'all' (default: sst2)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "stage3", "stage4"],
        default="auto",
        help="Which checkpoint to extract from. 'auto' (default) prefers "
             "stage4_range_aware/best_model when it exists, else staged_lpan_final.",
    )
    args = parser.parse_args()

    models = MODEL_KEYS if "all" in args.model else args.model
    tasks = TASK_KEYS if "all" in args.task else args.task
    for m in models:
        if m not in MODEL_KEYS:
            parser.error(f"unknown model {m!r}; options: {MODEL_KEYS}")
    for t in tasks:
        if t not in TASK_KEYS:
            parser.error(f"unknown task {t!r}; options: {TASK_KEYS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for name in models:
            if args.source == "stage3":
                model_dir = _ckpt_path(name, task, prefer_stage4=False)
            elif args.source == "stage4":
                base = Path("results/multi_model") / task / name
                model_dir = base / "stage4_range_aware" / "best_model"
            else:
                model_dir = _ckpt_path(name, task, prefer_stage4=True)
            if not model_dir.exists():
                print(f"  SKIP {name}/{task}: {model_dir} not found")
                continue

            coeffs = extract(str(model_dir))
            out_path = _out_path(name, task)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(coeffs, f, indent=2)

            n_gelu = sum(1 for v in coeffs.values() if v["activation_type"] == "gelu")
            n_ln = sum(1 for v in coeffs.values() if v["activation_type"] == "layernorm")
            n_soft = sum(1 for v in coeffs.values() if v["activation_type"] == "softmax")
            degrees = sorted(set(v["degree"] for v in coeffs.values()))

            print(
                f"  BERT-{name.capitalize()} × {task}: {len(coeffs)} polynomials "
                f"(GELU={n_gelu}, LN={n_ln}, Softmax={n_soft}), "
                f"degrees={degrees} → {out_path}\n"
                f"      source: {model_dir}"
            )


if __name__ == "__main__":
    main()
