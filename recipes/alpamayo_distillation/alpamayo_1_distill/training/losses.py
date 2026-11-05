"""Loss composition shared by the production loop and regression tests."""

from collections.abc import Mapping

import torch


def combine_training_loss(
    fallback_loss: torch.Tensor,
    loss_dict: Mapping[str, torch.Tensor] | None,
    *,
    future_traj_weight: float,
    future_traj_vlm_weight: float,
    distill_weight: float,
    vlm_loss_weight: float,
    on_policy_kl_weight: float,
    hidden_distill_weight: float,
) -> torch.Tensor:
    """Reproduce the production weighted-loss contract.

    When all explicitly composed objectives use their historical defaults, the
    model-provided aggregate is returned unchanged. Otherwise the individual
    terms are combined with exactly the weights used by the pre-refactor loop.
    """

    use_explicit_weights = (
        future_traj_weight != 1.0
        or future_traj_vlm_weight != 1.0
        or distill_weight != 1.0
        or vlm_loss_weight != 1.0
        or hidden_distill_weight != 0.0
    )
    if not use_explicit_weights or loss_dict is None:
        return fallback_loss

    weighted_loss = (
        future_traj_weight * loss_dict["future_traj_loss"]
        + future_traj_vlm_weight * loss_dict.get("future_traj_vlm_loss", 0.0)
        + distill_weight * loss_dict["distill_loss"]
        + vlm_loss_weight * loss_dict["vlm_loss"]
        + on_policy_kl_weight * loss_dict.get("on_policy_kl", 0.0)
    )
    if hidden_distill_weight > 0 and "hidden_distill_loss" in loss_dict:
        weighted_loss = weighted_loss + (
            hidden_distill_weight * loss_dict["hidden_distill_loss"]
        )
    return weighted_loss
