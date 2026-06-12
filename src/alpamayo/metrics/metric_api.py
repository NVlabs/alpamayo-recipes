# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Mapping, MutableMapping
import torch

from alpamayo_r1.common import logging
from alpamayo_r1.geometry import rotation
from alpamayo.metrics.metric_utils import apply_prefix
from alpamayo.metrics import collision_metrics, distance_metrics

# default vehicle size in meters
EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)

logger = logging.RankedLogger(__name__, rank_zero_only=False)
logger.setLevel("INFO")


class Metric(ABC):
    """Base class for metrics, subclass and implement the evaluate function."""

    @abstractmethod
    def evaluate(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluates metric(s) on a batch of data.

        Args:
            model (BaseModel): the model being trained
            data_batch (dict[str,torch.Tensor]): batch of data
            output_batch (dict[str,torch.Tensor]): model outputs from model.training_step()
                or model.validation_step()

        Returns:
            per_sample_metrics (dict[str,torch.Tensor]): metrics computed per sample where each
                key is a metric name and the value is the metric value of size [B]
        """
        ...


class ReasoningSampler(Metric):
    """Helper metric for use in `alpamayo.callbacks.metric_callback.MetricRunnerCallback` which
    samples reasoning process from the model and adds them to output_batch for use in later metrics
    and vis.

    Does not compute any actual metric values.
    """

    def __init__(
        self,
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        prefix: str = "",
        max_generation_length: int = 256,
        traj_only_generation: bool = False,
        **kwargs,
    ) -> None:
        """Reasoning sampler module.

        Args:
            top_p (float, optional): top probability to sample. Defaults to 0.98.
            temperature (float, optional): sampling temperature. Defaults to 1.0.
            num_traj_samples (int, optional): number of trajectory samples per set. Defaults to 6.
            num_traj_sets (int, optional): number of trajectory sets, used to compute variance.
                Defaults to 1.
            prefix (str, optional): prefix to attach to the predictions which are added to the
                output_batch. Defaults to "".
            kwargs (Mapping[str, object], optional): additional arguments to pass to
                `model.sample_trajectories_from_data`.
        """
        super().__init__()
        self.top_p = top_p
        self.temperature = temperature
        self.num_traj_samples = num_traj_samples
        self.num_traj_sets = num_traj_sets
        self.prefix = prefix
        self.max_generation_length = max_generation_length
        self.kwargs = kwargs if kwargs is not None else {}

    def evaluate(
        self,
        model,
        data_batch: Mapping[str, torch.Tensor],
        output_batch: MutableMapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Generates predictions from the model and adds them to the output_batch.

        Outputs dict contains:
            pred_xyz: [B, N, K, Tf, 3] predicted trajectory
            pred_rot: [B, N, K, Tf, 3, 3] predicted rotations
            logprob: [B, N, K, Tf] log probabilities of predicted tokens
            cot: [B, ns, nj] predicted Cot with number of set (ns) and number of traj (nj)
            meta_action_string: [B, ns, nj] predicted meta action strings
            pred_answer: [B, ns, nj] predicted answers
        """
        pred_xyz, pred_rot = model.sample_trajectories_from_data(
            data=data_batch,
            num_traj_samples=self.num_traj_samples,
            num_traj_sets=self.num_traj_sets,
            top_p=self.top_p,
            temperature=self.temperature,
            traj_only_generation=False,
            max_generation_length=self.max_generation_length,
            return_extra=False,
            **self.kwargs,
        )

        # dict used for later metrics
        output_batch.update(
            apply_prefix(
                self.prefix,
                {
                    "pred_xyz": pred_xyz,
                    "pred_rot": pred_rot,
                },
            )
        )

        # dict for data input (no need to add prefix)
        output_batch.update(
            {
                "absolute_timestamps": data_batch.get("absolute_timestamps", None),
                "relative_timestamps": data_batch.get("relative_timestamps", None),
                "ego_history_xyz": data_batch.get("ego_history_xyz", None),
                "ego_history_rot": data_batch.get("ego_history_rot", None),
                "ego_future_xyz": data_batch.get("ego_future_xyz", None),
                "ego_future_rot": data_batch.get("ego_future_rot", None),
            }
        )
        return {}


class DistanceMetrics(Metric):
    """Computes distance metrics for `alpamayo.callbacks.metric_callback.MetricRunnerCallback`."""

    def __init__(
        self,
        prefix: str = "",
        time_step: float = 0.1,
    ):
        """Args:
        prefix (str, optional): prefix of trajectory samples to use. Computes metrics on
            trajectories in output_batch with keys f"{prefix}pred_rot"
            Defaults to "".
        group_by_scenarios (list[str], optional): Defaults to None.
            if not None, then the metrics will be grouped by the scenario names in this list.
        """
        self.prefix = prefix
        self.time_step = time_step

    def evaluate(
        self,
        model: torch.nn.Module | None,
        data_batch: Mapping[str, torch.Tensor],
        output_batch: MutableMapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Computes distance metrics (corner distance, minADE, ADE)

        Assumes TrajSampler has been run already, and trajectory samples are
        in the output_batch with keys:
            (prefix)+pred_xyz: [B, N, K, Tf, 3] predicted trajectory
            (prefix)+pred_rot [B, N, K, Tf, 3, 3] predicted rotations
            (prefix)+logprob: [B, N, K, Tf] log probabilities of predicted tokens

        Returns dict[str,Tensor]:
            # distance metrics
            min_ade: [B] average min_ade (min over K, average over N)
            corner_distance: [B] average min corner distance (min over K, average over N)

            # if N > 1, we have the following extra data item:
            # for each value above, we collect the statistics _sq and _std, for example:
            min_ade_sq: [B] average min_ade^2 (min over K, average over N)
            min_ade_std: [B] std of min_ade
        """
        del model  # unused

        if self.prefix + "pred_xyz" not in output_batch:
            logger.warning(f"No predictions with prefix {self.prefix} found in output_batch.")
            return {}

        pred_xyz = output_batch[self.prefix + "pred_xyz"]
        pred_rot = output_batch[self.prefix + "pred_rot"]

        num_traj_sets = pred_xyz.shape[1]

        if data_batch["ego_future_xyz"].shape[1] > 1:
            logger.info("Multiple traj group provided, only evaluating the last one.")
        gt_xyz = data_batch["ego_future_xyz"][:, -1]
        gt_rot = data_batch["ego_future_rot"][:, -1]

        # TODO: move this to a config
        timestep_horizons_in_seconds = [0.5, 1, 3, 5]
        timestep_horizons = [int(t / self.time_step) for t in timestep_horizons_in_seconds]

        metric_dict = distance_metrics.compute_minade(
            pred_xyz,
            gt_xyz,
            disable_summary=(num_traj_sets == 1),
            timestep_horizons=timestep_horizons,
            time_step=self.time_step,
        )

        # compute per-sample ADE for later visualization
        sample_ade = distance_metrics.compute_ade(pred_xyz, gt_xyz)
        output_batch.update({self.prefix + "sample_ade": sample_ade})

        # dummy logprob for now
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # compute ADE, select pred_xyz of highest logprob
        # logprob: [B, N, K, Tf]
        idx = logprob.sum(dim=-1).argmax(dim=-1)  # [B, N]
        top_xyz = torch.take_along_dim(pred_xyz, idx[..., None, None, None], dim=2)
        ade = distance_metrics.compute_ade(top_xyz, gt_xyz).squeeze(2).mean(-1)  # [B]
        metric_dict.update({"ade": ade})
        # shape [B]
        timestep_horizon = int(3.0 / self.time_step)
        if timestep_horizon <= pred_xyz.shape[3]:
            ade_3s = (
                distance_metrics.compute_ade(top_xyz, gt_xyz, timestep_horizon=timestep_horizon)
                .squeeze(2)
                .mean(-1)
            )
            metric_dict.update({"ade/by_t=3.0": ade_3s})

        metric_dict.update(
            distance_metrics.compute_grouped_corner_distance(
                pred_xyz,
                pred_rot,
                gt_xyz,
                gt_rot,
                # TODO: change this to true ego vehicle size
                torch.tensor(EGO_VEHICLE_LWH, dtype=torch.float32, device=gt_xyz.device),
                disable_summary=(num_traj_sets == 1),
            )
        )

        return apply_prefix(self.prefix, metric_dict)


class CollisionMetrics(Metric):
    """Computes 2D collision metrics against obstacles using OBB edge-point sampling."""

    def __init__(self, prefix: str = "", timestep_horizons: list[int] | None = None) -> None:
        """Initialize collision metrics evaluator.

        Args:
            prefix: Prefix for trajectory keys in output_batch.
            timestep_horizons: Future timestep indices at which to evaluate collisions.
                Must be positive. Defaults to [1, 5, 10, 25, 50].
        """
        super().__init__()
        self.prefix = prefix
        if timestep_horizons is None:
            timestep_horizons = [1, 5, 10, 25, 50]
        for t in timestep_horizons:
            if t <= 0:
                raise ValueError("Requested timestep_horizons must all be positive!")
        self.timestep_horizons = timestep_horizons

    def evaluate(
        self,
        model: torch.nn.Module | None,
        data_batch: Mapping[str, torch.Tensor],
        output_batch: MutableMapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute collision metrics.

        Requires ``data_batch`` to contain:
            - ``ego_lwh``: [B, 3]
            - ``ego_length_offset``: [B]
            - ``ego_history_xyz``: [B, 1, Th, 3]
            - ``ego_history_rot``: [B, 1, Th, 3, 3]
            - ``obstacle_bbox_history``: [B, 1, N_obs, Th, 7] (xyz, lwh, yaw)
            - ``obstacle_bbox_future``: [B, 1, N_obs, Tf, 7]

        Requires ``output_batch`` to contain:
            - ``{prefix}pred_xyz``: [B, N, K, Tf, 3]
            - ``{prefix}pred_rot``: [B, N, K, Tf, 3, 3]

        Returns:
            Dict with ``collision/by_t=X.X`` keys mapping to [B] float tensors.
        """
        del model  # unused
        if "obstacle_bbox_future" not in data_batch:
            logger.warning("No obstacle future bounding boxes found in data_batch.")
            return {}
        if self.prefix + "pred_xyz" not in output_batch:
            logger.warning(f"No predictions with prefix {self.prefix} found in output_batch.")
            return {}

        if data_batch["ego_future_xyz"].shape[1] > 1:
            logger.info("Multiple traj group provided, only evaluating the last one.")

        obstacle_xyz_lwh_yaw = data_batch["obstacle_bbox_history"][:, -1, :, -1, :]
        obstacle_box = collision_metrics.BoxObstacle(
            obstacle_xyz_lwh_yaw[:, :, 0],
            obstacle_xyz_lwh_yaw[:, :, 1],
            obstacle_xyz_lwh_yaw[:, :, -1],
            obstacle_xyz_lwh_yaw[:, :, 3],
            obstacle_xyz_lwh_yaw[:, :, 4],
        )

        ego_xyz = data_batch["ego_history_xyz"][:, -1, -1, :]
        ego_rot = data_batch["ego_history_rot"][:, -1, -1, :, :]
        ego_heading = rotation.so3_to_yaw_torch(ego_rot)

        ego_length = data_batch["ego_lwh"][:, 0]
        ego_width = data_batch["ego_lwh"][:, 1]
        ego_length_offset = data_batch["ego_length_offset"]

        ego_box = collision_metrics.BoxObstacle(
            x=ego_xyz[:, None, 0],
            y=ego_xyz[:, None, 1],
            heading=ego_heading[:, None],
            length=ego_length[:, None],
            width=ego_width[:, None],
            length_offset=ego_length_offset[:, None],
        )

        current_collisions = ego_box.in_collision(obstacle_box)
        current_collisions = torch.where(current_collisions.isnan(), False, current_collisions)
        current_in_collision_with_any = current_collisions.any(dim=1)

        obstacle_xyz_lwh_yaw = data_batch["obstacle_bbox_future"][:, -1]
        obstacle_box = collision_metrics.BoxObstacle(
            x=obstacle_xyz_lwh_yaw[:, :, None, :, 0],
            y=obstacle_xyz_lwh_yaw[:, :, None, :, 1],
            heading=obstacle_xyz_lwh_yaw[:, :, None, :, -1],
            length=obstacle_xyz_lwh_yaw[:, :, None, :, 3],
            width=obstacle_xyz_lwh_yaw[:, :, None, :, 4],
        )

        ego_xyz_pred = output_batch[self.prefix + "pred_xyz"]
        ego_rot_pred = output_batch[self.prefix + "pred_rot"]
        ego_heading_pred = rotation.so3_to_yaw_torch(ego_rot_pred)

        gt_xyz = data_batch["ego_future_xyz"][:, -1, None, None, :, :]
        ade = (ego_xyz_pred - gt_xyz).pow(2).sum(-1).sqrt().mean(-1)
        min_ade_index = ade.argmin(-1)

        closest_ego_xyz = torch.take_along_dim(
            ego_xyz_pred, min_ade_index[:, :, None, None, None], dim=2
        ).squeeze(2)
        closest_ego_heading = torch.take_along_dim(
            ego_heading_pred, min_ade_index[:, :, None, None], dim=2
        ).squeeze(2)

        ego_box = collision_metrics.BoxObstacle(
            x=closest_ego_xyz[..., 0].unsqueeze(1),
            y=closest_ego_xyz[..., 1].unsqueeze(1),
            heading=closest_ego_heading.unsqueeze(1),
            length=ego_length[:, None, None, None],
            width=ego_width[:, None, None, None],
            length_offset=ego_length_offset[:, None, None, None],
        )

        collisions = ego_box.in_collision(obstacle_box)
        collisions = torch.where(collisions.isnan(), False, collisions)
        in_collision_with_any = collisions.any(dim=1)

        output: dict[str, torch.Tensor] = {
            "collision/by_t=0.0": current_in_collision_with_any.float(),
        }
        for t in self.timestep_horizons:
            output[f"collision/by_t={t / 10:0.1f}"] = (
                in_collision_with_any[:, :, :t].any(dim=-1).float().mean(dim=1)
            )

        return output
