#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode a board run of one clip and compare it with the reference.

Usage: python compare.py CLIP_DIR RESULT_DIR MODEL_CONFIG
  CLIP_DIR    from host/make_inputs.py (clip.json with the ego history, reference/)
  RESULT_DIR  from thor/run.py (output.json with CoC text + (accel, kappa) pairs)

Writes RESULT_DIR/compare.json: decoded xyz trajectories (t0 rig frame, 64 x 0.1 s),
the formatted prompt check against the reference prompt, CoC text versus the reference
CoCs, and pairwise ADE between the candidates.

The decode is the Alpamayo2-Edge UnicycleAccelCurvatureCourseActionSpace: normalized
(accel, curvature) -> speed and course integrated with the trapezoid rule from a t0
speed and course fitted on the ego history.
"""

import json
import re
import sys
from pathlib import Path

import torch

FADE = (0.5, 1.0)  # m/s: course offset fades in between these speeds


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def _unwrap(phi):
    d = _wrap(torch.diff(phi, dim=-1))
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def _yaw(rot):
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


class CourseActionSpace:
    def __init__(self, cfg: dict):
        self.accel_mean, self.accel_std = cfg["accel_mean"], cfg["accel_std"]
        self.curvature_mean, self.curvature_std = cfg["curvature_mean"], cfg["curvature_std"]
        self.slip_gradient, self.dt = cfg["slip_gradient"], cfg["dt"]
        self.v_lambda, self.v_ridge = cfg["v_lambda"], cfg["v_ridge"]

    def t0_speed(self, xyz, rot):
        """Trapezoidal-velocity least squares over the history (3rd-order smoothing, ridge)."""
        dxy = xyz[1:, :2] - xyz[:-1, :2]
        theta = _unwrap(_yaw(rot))
        n, dt = dxy.shape[0], self.dt
        g = 2 / dt * dxy
        a = torch.zeros(2 * n, n + 1, dtype=dxy.dtype)
        c, s, r = torch.cos(theta), torch.sin(theta), torch.arange(n)
        a[2 * r, r], a[2 * r, r + 1] = c[:-1], c[1:]
        a[2 * r + 1, r], a[2 * r + 1, r + 1] = s[:-1], s[1:]
        d = torch.zeros(max(n - 2, 0), n + 1, dtype=dxy.dtype)
        rr = torch.arange(d.shape[0])
        d[rr, rr], d[rr, rr + 1], d[rr, rr + 2], d[rr, rr + 3] = -1.0, 3.0, -3.0, 1.0
        lhs = (
            a.T @ a
            + (self.v_lambda / dt**6) * (d.T @ d)
            + self.v_ridge * torch.eye(n + 1, dtype=dxy.dtype)
        )
        v = torch.cholesky_solve((a.T @ g.flatten())[:, None], torch.linalg.cholesky(lhs))[:, 0]
        return v[-1]

    def t0_course(self, xyz, rot):
        """Course of the last history chord relative to the mean yaw of the last two poses, faded at low speed."""
        yaw = _unwrap(_yaw(rot[-2:])).mean()
        chord = xyz[-1, :2] - xyz[-2, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        along = c * chord[0] + s * chord[1]
        across = c * chord[1] - s * chord[0]
        offset = torch.atan2(across * torch.sign(along), along.abs())
        speed = chord.norm() / self.dt
        return offset * ((speed - FADE[0]) / (FADE[1] - FADE[0])).clamp(0.0, 1.0)

    def decode(self, action, xyz, rot):
        """action [..., 64, 2] normalized (accel, curvature) -> xyz [..., 64, 3] in the t0 rig frame."""
        dt = self.dt
        accel = action[..., 0] * self.accel_std + self.accel_mean
        kappa = action[..., 1] * self.curvature_std + self.curvature_mean
        v0 = self.t0_speed(xyz, rot).expand(accel.shape[:-1])
        vel = torch.cat([v0[..., None], v0[..., None] + torch.cumsum(accel * dt, -1)], -1)
        theta0 = self.t0_course(xyz, rot).expand(accel.shape[:-1])[..., None]
        theta = torch.cat(
            [
                theta0,
                theta0
                + torch.cumsum(kappa * vel[..., :-1] * dt, -1)
                + torch.cumsum(kappa * accel * (0.5 * dt**2), -1),
            ],
            -1,
        )
        h = 0.5 * dt
        x = torch.cumsum(vel[..., :-1] * torch.cos(theta[..., :-1]) * h, -1) + torch.cumsum(
            vel[..., 1:] * torch.cos(theta[..., 1:]) * h, -1
        )
        y = torch.cumsum(vel[..., :-1] * torch.sin(theta[..., :-1]) * h, -1) + torch.cumsum(
            vel[..., 1:] * torch.sin(theta[..., 1:]) * h, -1
        )
        return torch.stack([x, y, xyz[-1, 2].expand_as(x)], -1)


def main(clip_dir: Path, result_dir: Path, model_config: Path) -> None:
    clip = json.load(open(clip_dir / "clip.json"))
    responses = json.load(open(result_dir / "output.json"))["responses"]
    cfg = json.load(open(model_config))["expert_config"]["action_space_cfg"]
    space = CourseActionSpace(cfg)
    xyz, rot = torch.tensor(clip["ego_history_xyz"]), torch.tensor(clip["ego_history_rot"])
    actions = torch.tensor(
        [r["output_trajectory"] for r in responses], dtype=torch.float32
    )  # [B, 64, 2]
    traj = space.decode(actions, xyz, rot)  # [B, 64, 3]
    pair_ade = (traj[:, None, :, :2].double() - traj[None, :, :, :2].double()).norm(dim=-1).mean(-1)

    report = {
        "clip": clip_dir.name,
        "candidates": len(responses),
        "t0_speed_mps": round(space.t0_speed(xyz, rot).item(), 3),
        # generation stops one token after <|traj_future_start|> (the expert's KV ends at that marker), so the
        # text carries the first future-trajectory token; keep the chain-of-causation sentence only.
        "coc": [r["output_text"].split("\n")[0].strip() for r in responses],
        "raw_output_text": [r["output_text"] for r in responses],
        "trajectory_xyz": traj.tolist(),
        "pairwise_ade_m": pair_ade.tolist(),
        "endpoint_xy_m": traj[:, -1, :2].tolist(),
    }
    profile = result_dir / "profile.json"
    if profile.exists():
        profile = json.load(open(profile))
        if "prompt_tokens" in profile:  # thor/run.py
            report["prompt_tokens"] = profile["prompt_tokens"]
            report["stages_ms"] = profile["stages_ms"]
            report["total_ms"] = profile["total_ms"]
        else:  # stock action_inference --dump-profile
            report["prompt_tokens"] = profile["prefill"]["average_tokens_per_run"] / len(responses)
    ref = clip_dir / "reference"
    if ref.exists():
        # The runtime expands one <|image_pad|> per image into the image's tokens, so compare with runs collapsed.
        collapse = lambda s: re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", s.strip())
        prompt = collapse((ref / "prompt.txt").read_text())
        # The runner fills the <|traj_history|> placeholders from the trajectory item; fill them the same way here.
        # thor/run.py has no formatted prompt in its output: it ran the prompt.json text tokenized on the host.
        ours = (
            responses[0].get("formatted_complete_request")
            or json.load(open(clip_dir / "prompt.json"))["text"]
        )
        ours = collapse(ours).replace(
            "<|traj_history|>" * len(clip["history_tokens"]),
            "".join(f"<i{t}>" for t in clip["history_tokens"]),
        )
        report["prompt_matches_reference"] = prompt == ours
        coc_ref = json.load(open(ref / "coc.json"))
        report["prompt_tokens_match_reference"] = (
            report.get("prompt_tokens") == coc_ref["prompt_tokens"]
        )
        if prompt != ours:
            i = next(
                (k for k, (a, b) in enumerate(zip(prompt, ours)) if a != b),
                min(len(prompt), len(ours)),
            )
            report["prompt_first_difference"] = {
                "at": i,
                "reference": prompt[max(0, i - 60) : i + 60],
                "ours": ours[max(0, i - 60) : i + 60],
            }
        strip = lambda c: c.replace("<think></think>", "").strip()
        report["reference_coc"] = [strip(c) for c in coc_ref["coc"]]
        report["reference_sampling"] = coc_ref["sampling"]
        report["coc_in_reference_set"] = [
            strip(c) in set(report["reference_coc"]) for c in report["coc"]
        ]
    json.dump(report, open(result_dir / "compare.json", "w"), indent=1)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("trajectory_xyz", "pairwise_ade_m")},
            indent=1,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(
            __doc__ + "\n  MODEL_CONFIG  the release config.json (shim/alpamayo2_edge.config.json)"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
