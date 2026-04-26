#!/usr/bin/env python3
"""Per-task pipeline driver.

Runs the *full* FHE-readiness pipeline for one (model, task) pair:

    1. LPAN train      (skip if checkpoint exists)
    2. Stage-4+KD      (skip if checkpoint exists)
    3. FHE audit       (always re-run; cheap)
    4. OpenFHE single FFN block  (validates encryption stack end-to-end)
    5. Block-level latency benchmark
    6. Per-task summary  → results/per_task_summary/<model>_<task>.json

Use --skip-lpan / --skip-s4 / --skip-encrypt to short-circuit stages.
Use --force to redo all stages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fhe_thesis.config import MODEL_REGISTRY, MULTI_MODEL_DIR, RESULTS_DIR  # noqa: E402

PYTHON = sys.executable
SUMMARY_DIR = RESULTS_DIR / "per_task_summary"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--task", required=True, choices=["sst2", "mrpc", "qnli", "qqp"])
    p.add_argument("--skip-lpan", action="store_true")
    p.add_argument("--skip-s4", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--skip-encrypt", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run_step(cmd: list[str], log_path: Path, label: str) -> tuple[int, float]:
    """Run a subprocess, stream to log, return (exit_code, wall_seconds)."""
    print(f"\n  ▶ {label}")
    print(f"      cmd: {' '.join(cmd)}")
    print(f"      log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": str(REPO)}
        )
    wall = time.perf_counter() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"      → {status} in {wall:.1f}s")
    return proc.returncode, wall


def lpan_ckpt(model: str, task: str) -> Path:
    if task == "sst2":
        return MULTI_MODEL_DIR / model / "staged_lpan_final" / "best_model"
    return MULTI_MODEL_DIR / model / task / "staged_lpan_final" / "best_model"


def stage4_ckpt(model: str, task: str) -> Path:
    if task == "sst2":
        return MULTI_MODEL_DIR / model / "stage4_range_aware" / "best_model"
    return MULTI_MODEL_DIR / model / f"{task}_stage4_range_aware" / "best_model"


def main() -> int:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    log_root = RESULTS_DIR / "logs" / "per_task" / f"{args.model}_{args.task}"
    summary: dict = {
        "model": args.model,
        "task": args.task,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {},
    }

    print("=" * 70)
    print(f"  Per-task pipeline   model={args.model}   task={args.task}")
    print("=" * 70)

    # ── 1. LPAN ────────────────────────────────────────────────────
    lpan = lpan_ckpt(args.model, args.task)
    if (lpan / "config.json").exists() and not args.force and not args.skip_lpan:
        print(f"\n  [LPAN]  ✓ checkpoint exists: {lpan}")
        summary["stages"]["lpan"] = {"status": "exists", "checkpoint": str(lpan)}
    elif args.skip_lpan:
        print("\n  [LPAN]  ⊘ skipped (--skip-lpan)")
        summary["stages"]["lpan"] = {"status": "skipped"}
    else:
        rc, wall = run_step(
            [PYTHON, "run_staged_lpan.py", "--model", args.model, "--task", args.task],
            log_root / "1_lpan.log",
            "LPAN curriculum",
        )
        summary["stages"]["lpan"] = {
            "status": "ok" if rc == 0 else "fail",
            "wall_s": wall,
        }
        if rc != 0:
            (SUMMARY_DIR / f"{args.model}_{args.task}.json").write_text(json.dumps(summary, indent=2))
            return rc

    # ── 2. Stage-4+KD ─────────────────────────────────────────────
    s4 = stage4_ckpt(args.model, args.task)
    if (s4 / "config.json").exists() and not args.force and not args.skip_s4:
        print(f"\n  [S4]    ✓ checkpoint exists: {s4}")
        summary["stages"]["stage4"] = {"status": "exists", "checkpoint": str(s4)}
    elif args.skip_s4:
        print("\n  [S4]    ⊘ skipped (--skip-s4)")
        summary["stages"]["stage4"] = {"status": "skipped"}
    else:
        rc, wall = run_step(
            [
                PYTHON, "run_stage4_range_aware.py",
                "--model", args.model, "--task", args.task,
                "--device", "cpu",
            ],
            log_root / "2_stage4.log",
            "Stage-4 range-aware FT + KD",
        )
        summary["stages"]["stage4"] = {
            "status": "ok" if rc == 0 else "fail",
            "wall_s": wall,
        }
        if rc != 0:
            (SUMMARY_DIR / f"{args.model}_{args.task}.json").write_text(json.dumps(summary, indent=2))
            return rc

    # ── 3. FHE-readiness audit ────────────────────────────────────
    if args.skip_audit:
        print("\n  [Audit] ⊘ skipped (--skip-audit)")
        summary["stages"]["audit"] = {"status": "skipped"}
    else:
        rc, wall = run_step(
            [
                PYTHON, "experiments/audit_fhe_readiness.py",
                "--model", args.model, "--task", args.task,
                "--num-samples", "256", "--device", "cpu",
            ],
            log_root / "3_audit.log",
            "FHE-readiness audit (post-Stage-4)",
        )
        audit_path = RESULTS_DIR / "fhe_readiness" / f"{args.task}_{args.model}.json"
        if audit_path.exists():
            audit_data = json.loads(audit_path.read_text())
            summary["stages"]["audit"] = {
                "status": "ok" if rc == 0 else "fail",
                "wall_s": wall,
                "max_oor_pct": max((100 * v["oor_frac"] for v in audit_data.get("ops", {}).values()), default=0.0),
                "max_excursion": max((v["max_excursion"] for v in audit_data.get("ops", {}).values()), default=0.0),
            }
        else:
            summary["stages"]["audit"] = {"status": "fail", "wall_s": wall}

    # ── 4. OpenFHE single-block smoke ─────────────────────────────
    if args.skip_encrypt:
        print("\n  [Encrypt] ⊘ skipped (--skip-encrypt)")
        summary["stages"]["encrypt"] = {"status": "skipped"}
    else:
        rc, wall = run_step(
            [
                PYTHON, "experiments/run_e2e_openfhe.py",
                "--model", args.model, "--task", args.task,
                "--seq-len", "8",
                "--no-bootstrap", "--no-classifier",
                "--mult-depth", "25", "--ring-dim", "16384",
            ],
            log_root / "4_encrypt.log",
            "OpenFHE encrypted forward (no bootstrap, smoke)",
        )
        e2e_path = RESULTS_DIR / "encrypted_inference" / f"{args.model}_{args.task}_e2e_openfhe.json"
        if e2e_path.exists():
            ed = json.loads(e2e_path.read_text())
            summary["stages"]["encrypt"] = {
                "status": "ok" if rc == 0 else "fail",
                "wall_s": wall,
                "total_compute_s": ed.get("total_compute_sec"),
                "per_layer_compute_s": ed.get("per_layer_compute_sec"),
            }
        else:
            summary["stages"]["encrypt"] = {"status": "fail", "wall_s": wall}

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = SUMMARY_DIR / f"{args.model}_{args.task}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary → {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
