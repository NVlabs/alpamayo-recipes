# Action Expert Model Support for AlpaGym

This package contains model-side support for closed-loop reinforcement
learning of the Alpamayo diffusion action expert through AlpaGym.

The code in this directory was added so AlpaGym can import the action expert
Cosmos-RL wrapper, SDE-aware flow-matching diffusion utilities, and GRPO
log-probability replay logic from `alpamayo1_x_rl` instead of carrying a
separate vendored copy.

## Scope

This directory is intended for the AlpaGym closed-loop RL integration.

It is not a standalone action expert RL recipe in `alpamayo-recipes`. In
particular, this package does not add user-facing instructions for running
open-loop action expert RL from this repository.

Existing `alpamayo1_x_rl` users should continue to treat the documented
recipe entrypoints under `models/reasoning_vla/` as the supported open-loop
VLM RL workflow. The top-level `alpamayo1_x_rl` README remains the source of
truth for the currently documented recipe commands.

## Integration Boundary

The expected boundary is:

- `alpamayo_r1`: released Alpamayo model primitives
- `alpamayo1_x_rl.models.expert_model`: importable action expert model and
  diffusion support for AlpaGym
- AlpaGym: closed-loop environment interaction, orchestration, and launch
  integration
