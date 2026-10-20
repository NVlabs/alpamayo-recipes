"""Base processor interface for VLM backends."""

from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseVLMProcessor(ABC):
    """Abstract base class for VLM processors.

    This interface allows easy swapping between different VLM backends
    like Qwen, InternVL, etc.
    """

    def __init__(
        self,
        vlm_name_or_path: str,
        traj_vocab_size: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
    ) -> None:
        """Initialize the processor.

        Args:
            vlm_name_or_path: Path to the VLM model.
            traj_vocab_size: Size of the trajectory vocabulary.
            min_pixels: Minimum number of pixels for image resizing.
            max_pixels: Maximum number of pixels for image resizing.
            include_camera_ids: Whether to include camera IDs as text before images.
        """
        self.vlm_name_or_path = vlm_name_or_path
        self.traj_vocab_size = traj_vocab_size
        self._processor = None
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums

    @abstractmethod
    def build_processor(self) -> Any:
        """Build and return the VLM processor."""
        pass

    @abstractmethod
    def get_preprocess_data_fn(
        self,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Get the data preprocessing function."""
        pass

    @abstractmethod
    def collate_fn(self, data: list[dict[str, Any]], padding_side: str = "left") -> dict[str, Any]:
        """Collate function for batching data."""
        pass

    @property
    def processor(self) -> Any:
        """Get the vlm processor with expanded vocabulary for single data."""
        if self._processor is None:
            self._processor = self.build_processor()
        return self._processor
