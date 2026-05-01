"""High-level training pipelines built on top of :mod:`fhe_thesis.training`.

The flagship pipeline is :class:`HyperLPANPipeline`, which orchestrates the
full HyPER-LPAN training trajectory in a single resumable, config-driven run.
"""

from fhe_thesis.pipelines.hyper_lpan import (
    HyperLPANConfig,
    HyperLPANPipeline,
)

__all__ = ["HyperLPANConfig", "HyperLPANPipeline"]
