"""Benchmark harness for HyPER-LPAN thesis evaluation.

Modules
-------
accuracy : plaintext GLUE accuracy + multi-checkpoint comparison tables.
"""

from fhe_thesis.benchmarks.accuracy import (
    evaluate_checkpoint,
    compare_checkpoints,
)
from fhe_thesis.benchmarks.latency import (
    LatencyResult,
    aggregate_timings,
    profile_latency,
)

__all__ = [
    "evaluate_checkpoint",
    "compare_checkpoints",
    "LatencyResult",
    "aggregate_timings",
    "profile_latency",
]
