"""Camera rendering."""

from collections.abc import Sequence

import torch
from torch import Tensor

from scene_reconstruction.core import Volume
from scene_reconstruction.core.transform import einsum_transform


def depth_centers(min_depth: float, max_depth: float, num_bins: int):
    """Depth bins center points."""
    return ((torch.arange(num_bins) + 0.5) / num_bins) * (max_depth - min_depth) + min_depth


def camera_sample_grid(
    camera_volume_shape: Sequence[int], camera_from_image: Tensor, min_depth, max_depth, img_width, img_height
):
    """Create sample grid from camera intrinsics."""
    cam_lower = torch.stack([torch.zeros_like(img_width), torch.zeros_like(img_height), torch.ones_like(img_height)], 1)
    cam_upper = torch.stack([img_width, img_height, torch.ones_like(img_height)], 1)
    img_volume = Volume(lower=cam_lower, upper=cam_upper)

    img_grid = img_volume.new_coord_grid([camera_volume_shape[0], camera_volume_shape[1], 1])
    img_grid_camera_no_depth = einsum_transform("bci,bxyzi->bxyzc", camera_from_image, points=img_grid)
    img_grid_direction = img_grid_camera_no_depth / img_grid_camera_no_depth.norm(dim=-1, keepdim=True)
    img_grid_camera = img_grid_direction * depth_centers(min_depth, max_depth, camera_volume_shape[2]).view(
        1, 1, 1, -1, 1
    )
    return img_grid_camera


def sample_camera_rays(
    ego_features: Tensor,
    volume_ego: Volume,
    ego_from_camera: Tensor,
    camera_from_image: Tensor,
    camera_volume_shape: Sequence[int],
    min_depth,
    max_depth,
    img_width,
    img_height,
):
    """Sample camera rays in ego volume."""
    img_grid_camera = camera_sample_grid(
        camera_volume_shape=camera_volume_shape,
        camera_from_image=camera_from_image,
        min_depth=min_depth,
        max_depth=max_depth,
        img_width=img_width,
        img_height=img_height,
    )
    img_grid_ego = einsum_transform("bec,bxyzc->bxyze", ego_from_camera, points=img_grid_camera)
    # TODO: rescaling based on volume
    samped_features = volume_ego.sample_volume(ego_features, img_grid_ego)
    return samped_features
