"""Configuration for on-policy (student-sampled) distillation."""

from __future__ import annotations

from dataclasses import dataclass

from alpamayo1_distill.losses.divergence import DIVERGENCE_MODE_NAMES, FORWARD_KL

DIVERGENCE_MODES = DIVERGENCE_MODE_NAMES


@dataclass
class OnPolicyDistillConfig:
    """When and how to distil on the student's own rollouts.

    Off-policy distillation scores the student on sequences the *teacher* generated,
    so every prefix it ever sees is teacher-quality. At inference the student
    conditions on its own prefixes — including its own mistakes — and small errors
    compound. On-policy distillation closes that gap by sampling the sequence from
    the student and asking the teacher what it would have predicted there; it is the
    DAgger analogue of off-policy's behaviour cloning.

    Two consequences shape this config:

    * The hard CE term cannot come along — cross-entropy against tokens the student
      itself sampled just reinforces whatever it produced. On-policy steps supervise
      with the divergence only.
    * The student must already generate well-formed sequences (in particular, emit
      ``<traj_future_start>``), so mixing is normally enabled only after an
      off-policy warmup via :attr:`start_step`.
    """

    ratio: float = 0.0
    """Fraction of optimizer steps that use a student-sampled sequence. 0 disables."""

    start_step: int = 0
    """Global step at which mixing begins; steps before this stay fully off-policy."""

    divergence: str = FORWARD_KL
    """Divergence for on-policy steps — see :func:`~alpamayo1_distill.losses.token_divergence`."""

    beta: float = 0.5
    """Teacher mixture weight, used only when :attr:`divergence` is ``jsd``."""

    temperature: float = 1.0
    """Divergence softmax temperature. 1.0 matches the distribution actually sampled."""

    sample_temperature: float = 0.6
    """Sampling temperature for the student rollout. Defaults to the eval setting."""

    sample_top_p: float = 0.98
    """Nucleus cutoff for the student rollout. Defaults to the eval setting."""

    kl_weight: float = 1.0
    """Weight applied to the on-policy divergence in the total loss."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"ratio must lie in [0, 1], got {self.ratio}")
        if self.divergence not in DIVERGENCE_MODES:
            raise ValueError(
                f"divergence must be one of {DIVERGENCE_MODES}, got {self.divergence!r}"
            )
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")

    @property
    def enabled(self) -> bool:
        """Whether any step will be on-policy."""
        return self.ratio > 0.0

    @property
    def stride(self) -> int:
        """Take every ``stride``-th step on-policy.

        Interleaving one-in-N keeps the two regimes mixed throughout training rather
        than clustering them, which would let the student drift during a long
        on-policy block.
        """
        if not self.enabled:
            return 0
        return max(1, round(1.0 / self.ratio))

    @property
    def effective_ratio(self) -> float:
        """Realised fraction of on-policy steps after rounding to an integer stride."""
        return 0.0 if not self.enabled else 1.0 / self.stride

    def is_on_policy_step(self, global_step: int) -> bool:
        """Whether ``global_step`` should sample its sequence from the student.

        Derived purely from the step index so every rank agrees without
        communicating, and so a resumed run reproduces the same schedule.
        """
        if not self.enabled or global_step < self.start_step:
            return False
        return (global_step - self.start_step) % self.stride == 0

    def describe(self) -> str:
        """One-line summary for startup logs."""
        if not self.enabled:
            return "on-policy distillation: disabled"
        return (
            f"on-policy distillation: every {self.stride} step(s) "
            f"(effective ratio {self.effective_ratio:.3f}) from step {self.start_step}, "
            f"divergence={self.divergence}"
            + (f" beta={self.beta}" if self.divergence == "jsd" else "")
            + f", T={self.temperature}, sampling(temp={self.sample_temperature}, "
            f"top_p={self.sample_top_p}), kl_weight={self.kl_weight}"
        )
