from abc import ABC, abstractmethod

import torch


class BaseTrajectoryTokenizer(ABC):
    """Base class for trajectory tokenizers."""

    @property
    def vocab_size(self) -> int:
        """Tokens are integers from the set {0, 1, ..., vocab_size - 1}"""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Encodes the trajectories as discrete tokens. The model conditions on the historical
        waypoints to tokenize the future waypoints. Trajectories can be provided in any coordinate
        frame. Timestamps can be provided with any time-origin.

        Args:
            hist_xyz (torch.Tensor): Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot (torch.Tensor): Historical rotations. Shape: (B, Th, 3, 3).
            fut_xyz (torch.Tensor): Future locations XYZ. Shape: (B, Tf, 3).
            fut_rot (torch.Tensor): Future rotations. Shape: (B, Tf, 3, 3).
            hist_tstamp (torch.Tensor): Historical time stamps. Shape: (B, Th).
            fut_tstamp (torch.Tensor): Future time stamps. Shape: (B, Tf).

        Returns:
            torch.LongTensor: The token indices. Shape: (B, num_tokens_per_trajectory).
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def decode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        tokens: torch.LongTensor,
        hist_tstamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Decodes the given tokens into future trajectories. The future trajectory is returned in
        the same coordinate frame as the historical trajectory. Timestamps can be provided with any
        time-origin.

        Args:
            hist_xyz (torch.Tensor): Historical locations XYZ. Shape: (B, Th, 3).
            hist_rot (torch.Tensor): Historical rotations. Shape: (B, Th, 3, 3).
            tokens (torch.LongTensor): The token indices. Shape: (B, num_tokens_per_trajectory).
            hist_tstamp (torch.Tensor): Historical time stamps. Shape: (B, Th).

        Returns:
            fut_xyz (torch.Tensor): Future locations XYZ. Shape: (B, Tf, 3).
            fut_rot (torch.Tensor): Future rotations. Shape: (B, Tf, 3, 3).
            fut_tstamp (torch.Tensor): Future time stamps. Shape: (B, Tf).
        """
        raise NotImplementedError("Subclasses should implement this method.")
