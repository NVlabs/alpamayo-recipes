"""Loss functions for teacher-student distillation.

Public API — import from here rather than from the submodules, so call sites stay
stable if the internals move::

    from alpamayo1_distill.losses import (
        OnPolicyDistillConfig,
        masked_normalized_hidden_distance,
        sequence_distillation_loss,
        token_divergence,
    )

The package is deliberately free of model/trainer imports: everything here operates
on plain tensors so the functions can be unit-tested without building a VLM.
"""

from alpamayo1_distill.losses.config import DIVERGENCE_MODES, OnPolicyDistillConfig
from alpamayo1_distill.losses.divergence import sequence_distillation_loss, token_divergence
from alpamayo1_distill.losses.hidden import masked_normalized_hidden_distance

__all__ = [
    "DIVERGENCE_MODES",
    "OnPolicyDistillConfig",
    "masked_normalized_hidden_distance",
    "sequence_distillation_loss",
    "token_divergence",
]
