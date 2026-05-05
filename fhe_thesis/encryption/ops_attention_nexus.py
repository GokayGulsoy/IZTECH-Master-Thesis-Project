"""Backward-compatible re-export shim.

The original monolithic ``ops_attention_nexus.py`` has been split into
focused submodules during the production re-modularization. New code should
import from the focused modules directly:

- ``colmajor``  -- packing + key prep
- ``multi``     -- multi-ct arithmetic
- ``linear``    -- column-major linear projections (BSGS, streaming, multi)
- ``layernorm`` -- column-major LayerNorm (single + multi)
- ``attention`` -- Synthesizer-LPAN attention (Tay 2020 -> first FHE)

This shim is kept only to avoid breaking any in-tree script that still
spells the legacy module name. Slated for removal after one release cycle.
"""

from .colmajor import (  # noqa: F401
    pack_colmajor, unpack_colmajor,
    pack_colmajor_multi, unpack_colmajor_multi,
    prepare_colmajor_keys, _cols_per_ct,
)
from .multi import (  # noqa: F401
    add_multi, sub_multi, mul_multi, polyval_multi, per_col_sum_multi,
)
from .linear import (  # noqa: F401
    linear_colmajor,
    ColmajorLinearPlan, build_colmajor_linear_plan,
    ColmajorLinearPlanStreaming, build_colmajor_linear_plan_streaming,
    linear_colmajor_streaming,
    linear_colmajor_streaming_batched,
    linear_colmajor_multi_streaming_batched,
    linear_colmajor_multi_streaming,
    linear_colmajor_bsgs_cpp,
    per_col_sum_then_broadcast,
)
from .layernorm import (  # noqa: F401
    layernorm_colmajor, layernorm_colmajor_multi,
)
from .attention import (  # noqa: F401
    encode_synthesizer_diagonals, attn_synthesizer_nexus,
    encode_synthesizer_bsgs, attn_synthesizer_bsgs_nexus,
)
