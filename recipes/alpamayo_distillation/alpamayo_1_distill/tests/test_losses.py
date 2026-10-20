#!/usr/bin/env python3
"""Unit tests for alpamayo1_distill.losses.

Pure-tensor tests: no VLM, no dataset, no GPU required.

    python -m pytest alpamayo1_distill/tests/test_losses.py
    python -m alpamayo1_distill.tests.test_losses       # also runs standalone
"""

import math

import torch
import torch.nn.functional as F

from alpamayo1_distill.losses import (
    OnPolicyDistillConfig,
    masked_normalized_hidden_distance,
    sequence_distillation_loss,
    token_divergence,
)


# ---------------- token_divergence ----------------
def test_identical_distributions_have_zero_divergence():
    logits = torch.randn(32, 500)
    for mode in ("forward_kl", "reverse_kl", "jsd"):
        value = token_divergence(logits, logits.clone(), mode=mode)
        assert value.abs().item() < 1e-5, f"{mode} on identical inputs gave {value.item()}"


def test_divergences_are_non_negative():
    student, teacher = torch.randn(64, 300), torch.randn(64, 300)
    for mode in ("forward_kl", "reverse_kl", "jsd"):
        assert token_divergence(student, teacher, mode=mode).item() >= 0.0


def test_forward_kl_matches_reference_formula():
    """forward_kl must be KL(teacher || student), averaged per token."""
    student, teacher = torch.randn(16, 50), torch.randn(16, 50)
    got = token_divergence(student, teacher, mode="forward_kl")

    p_t = F.softmax(teacher, dim=-1)
    expected = (p_t * (F.log_softmax(teacher, dim=-1) - F.log_softmax(student, dim=-1))).sum(-1).mean()
    assert torch.allclose(got, expected, atol=1e-6), f"{got.item()} vs {expected.item()}"


def test_reverse_kl_matches_reference_formula():
    student, teacher = torch.randn(16, 50), torch.randn(16, 50)
    got = token_divergence(student, teacher, mode="reverse_kl")

    p_s = F.softmax(student, dim=-1)
    expected = (p_s * (F.log_softmax(student, dim=-1) - F.log_softmax(teacher, dim=-1))).sum(-1).mean()
    assert torch.allclose(got, expected, atol=1e-6), f"{got.item()} vs {expected.item()}"


def test_forward_and_reverse_kl_differ_and_are_asymmetric():
    """The whole point of offering both: they are not the same objective."""
    student = torch.tensor([[3.0, 1.0, -2.0], [0.0, 4.0, -1.0]])
    teacher = torch.tensor([[-1.0, 2.0, 1.0], [2.0, 0.0, -3.0]])
    fwd = token_divergence(student, teacher, mode="forward_kl")
    rev = token_divergence(student, teacher, mode="reverse_kl")
    assert not torch.allclose(fwd, rev, atol=1e-3)
    # reverse_kl(a, b) == forward_kl(b, a)
    swapped = token_divergence(teacher, student, mode="forward_kl")
    assert torch.allclose(rev, swapped, atol=1e-5)


def test_jsd_is_bounded_by_log2_and_symmetric_at_half():
    student, teacher = torch.randn(64, 200), torch.randn(64, 200)
    jsd = token_divergence(student, teacher, mode="jsd", beta=0.5)
    assert 0.0 <= jsd.item() <= math.log(2) + 1e-4, jsd.item()
    swapped = token_divergence(teacher, student, mode="jsd", beta=0.5)
    assert torch.allclose(jsd, swapped, atol=1e-5)


def test_gradient_flows_to_student_only():
    for mode in ("forward_kl", "reverse_kl", "jsd"):
        student = torch.randn(8, 40, requires_grad=True)
        teacher = torch.randn(8, 40, requires_grad=True)
        token_divergence(student, teacher, mode=mode).backward()
        assert student.grad is not None and student.grad.abs().sum() > 0, mode
        assert teacher.grad is None, f"{mode} leaked gradient into the teacher"


def test_temperature_softens_targets():
    student, teacher = torch.randn(32, 100), torch.randn(32, 100) * 3.0
    sharp = token_divergence(student, teacher, mode="forward_kl", temperature=0.5)
    soft = token_divergence(student, teacher, mode="forward_kl", temperature=4.0)
    assert soft.item() < sharp.item()


def test_rejects_bad_arguments():
    student, teacher = torch.randn(4, 10), torch.randn(4, 10)
    for bad in (lambda: token_divergence(student, torch.randn(4, 11)),
                lambda: token_divergence(student, teacher, temperature=0.0),
                lambda: token_divergence(student, teacher, mode="nope"),
                lambda: token_divergence(student, teacher, mode="jsd", beta=1.5)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---------------- sequence_distillation_loss ----------------
def test_sequence_loss_applies_causal_shift_and_mask():
    """Masked-out positions must not influence the result."""
    b, s, v = 2, 6, 30
    student, teacher = torch.randn(b, s, v), torch.randn(b, s, v)
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[:, 4:] = False

    base = sequence_distillation_loss(student, teacher, attention_mask=mask)
    # Perturb a single logit, not the whole row: softmax is shift-invariant, so
    # adding a constant across the vocabulary would change nothing anywhere.
    # Masked position 5 -> must not move the loss...
    student_perturbed = student.clone()
    student_perturbed[:, 5, 0] += 100.0
    assert torch.allclose(base, sequence_distillation_loss(student_perturbed, teacher, attention_mask=mask))
    # ...supervised position 2 -> must.
    student_perturbed = student.clone()
    student_perturbed[:, 2, 0] += 100.0
    assert not torch.allclose(base, sequence_distillation_loss(student_perturbed, teacher, attention_mask=mask))


def test_sequence_loss_matches_legacy_implementation():
    """Guards the refactor: identical numerics to the pre-refactor inline KL."""
    b, s, v = 2, 7, 40
    student, teacher = torch.randn(b, s, v), torch.randn(b, s, v)
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[0, 5:] = False
    T = 0.5

    got = sequence_distillation_loss(student, teacher, attention_mask=mask, temperature=T)

    # Verbatim legacy body.
    st, te = student[:, :-1, :], teacher[:, :-1, :]
    kd_mask = torch.ones(st.shape[:2], dtype=torch.bool) & (mask[:, 1:] > 0)
    s_logp = F.log_softmax(st[kd_mask] / T, dim=-1)
    t_p = F.softmax(te[kd_mask] / T, dim=-1)
    expected = F.kl_div(s_logp, t_p, reduction="batchmean")

    assert torch.allclose(got, expected, atol=1e-6), f"{got.item()} vs {expected.item()}"


def test_sequence_loss_returns_connected_zero_when_nothing_supervised():
    student = torch.randn(2, 5, 20, requires_grad=True)
    teacher = torch.randn(2, 5, 20)
    loss = sequence_distillation_loss(
        student, teacher, attention_mask=torch.zeros(2, 5, dtype=torch.bool)
    )
    assert loss.item() == 0.0
    loss.backward()  # must not raise: DDP needs a gradient path for every parameter
    assert student.grad is not None


def test_labels_select_supervised_positions():
    b, s, v = 2, 6, 25
    student, teacher = torch.randn(b, s, v), torch.randn(b, s, v)
    labels = torch.randint(0, v, (b, s))
    labels[:, :3] = -100
    with_labels = sequence_distillation_loss(student, teacher, labels=labels)
    mask_equivalent = sequence_distillation_loss(
        student, teacher, attention_mask=(labels != -100)
    )
    assert torch.allclose(with_labels, mask_equivalent, atol=1e-6)


# ---------------- hidden distance ----------------
def test_hidden_distance_zero_for_identical_and_two_for_opposite():
    hidden = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert masked_normalized_hidden_distance(hidden, hidden.clone(), mask).item() < 1e-6
    # antipodal unit vectors: 2 - 2*cos(pi) = 4
    assert abs(masked_normalized_hidden_distance(hidden, -hidden, mask).item() - 4.0) < 1e-4


# ---------------- schedule ----------------
def test_disabled_by_default():
    cfg = OnPolicyDistillConfig()
    assert not cfg.enabled
    assert not any(cfg.is_on_policy_step(step) for step in range(100))


def test_stride_realises_requested_ratio():
    for ratio, expected_stride in ((1.0, 1), (0.5, 2), (0.25, 4), (0.1, 10)):
        cfg = OnPolicyDistillConfig(ratio=ratio)
        assert cfg.stride == expected_stride, ratio
        hits = sum(cfg.is_on_policy_step(s) for s in range(1000))
        assert abs(hits / 1000 - ratio) < 0.01, (ratio, hits)


def test_start_step_gates_and_schedule_is_deterministic():
    cfg = OnPolicyDistillConfig(ratio=0.25, start_step=100)
    assert not any(cfg.is_on_policy_step(s) for s in range(100))
    assert cfg.is_on_policy_step(100)
    # Deterministic in the step index alone -> rank-consistent and resume-safe.
    other = OnPolicyDistillConfig(ratio=0.25, start_step=100)
    assert all(cfg.is_on_policy_step(s) == other.is_on_policy_step(s) for s in range(500))


def test_config_validates():
    for bad in (lambda: OnPolicyDistillConfig(ratio=1.5),
                lambda: OnPolicyDistillConfig(divergence="nope"),
                lambda: OnPolicyDistillConfig(start_step=-1)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    torch.manual_seed(0)
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} tests passed")
