# Util functions for data processing
import torch


def sort_images_by_camera_ids(
    image_frames: torch.Tensor, camera_indices: torch.Tensor, return_camera_ids: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sort camera images by their camera ids.

    Args:
        image_frames (torch.Tensor): shape (num_chunk, num_frames, 3, width, height)
        camera_indices (torch.Tensor): shape (num_chunk)
        return_camera_ids (bool): If True, also return the sorted camera indices

    Returns:
        torch.Tensor: Sorted image_frames tensor, arranged in ascending order of camera IDs.
        tuple[torch.Tensor, torch.Tensor]: If return_camera_ids is True,
            returns (sorted_image_frames, sorted_camera_indices).
    """
    sorted_indices = torch.argsort(camera_indices, stable=True)
    sorted_image_frames = image_frames[sorted_indices]
    if return_camera_ids:
        sorted_camera_indices = camera_indices[sorted_indices]
        return sorted_image_frames, sorted_camera_indices
    return sorted_image_frames


def num_batches(total_num: int, batch_size: int, drop_last: bool) -> int:
    """Get the number of batches for a given total number of items."""
    if drop_last:
        return total_num // batch_size
    else:
        return (total_num + batch_size - 1) // batch_size
