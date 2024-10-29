"""Dense spherical occupancy grid."""

import math
from collections.abc import Sequence
from typing import Literal, Optional, Union

import einops
import torch
from torch import Tensor

from scene_reconstruction.core import Volume
from scene_reconstruction.core.transform import einsum_transform
from scene_reconstruction.math.spherical_coordinate_system import cartesian_to_spherical, spherical_volume_element

MODES = Literal["clamp", "warp", "drop"]


def volume_density_from_points(
    volume_shape: Sequence[int],
    volume: Volume,
    points: Tensor,
    points_weights: Optional[Tensor] = None,
    modes: Sequence[Union[tuple[MODES, MODES], MODES]] = ("clamp", "clamp", "clamp"),
):
    """Scatters points into nearby voxels by considering surrounding voxels."""
    # points: [B, N, 3]
    # points_weights: [B, N, C]
    batch_size = points.shape[0]
    index_volume = Volume.new_index(volume_shape, device=points.device)
    index_from_volume = index_volume.self_from_other(volume)
    float_index = einsum_transform("biv,bnv->bni", index_from_volume, points=points)  # [B, N, 3]
    # next point on the voxel grid
    next_grid_index = float_index.round()
    # cube with side length 1.0 and center 0.5
    corner_offsets = 0.5 * torch.tensor(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=torch.float,
        device=volume.device,
    )
    cube_corners_float_index = float_index[:, :, None, :] + corner_offsets  # B, N, 8, 3
    # volume in each surrounding volume element
    corner_volume = (cube_corners_float_index - next_grid_index[:, :, None, :]).abs().prod(-1)  # B, N, 8
    cube_corners_index = cube_corners_float_index.floor().long()  # B, N, 8, 3
    assert len(modes) == cube_corners_index.shape[-1]
    cube_corners_index_processed = []
    for idx, mode, shape in zip(cube_corners_index.unbind(-1), modes, volume_shape):
        if not isinstance(mode, (list, tuple)):
            m_lower, m_upper = (mode, mode)
        else:
            m_lower, m_upper = mode

        lower_mask = idx < 0
        if m_lower == "clamp":
            idx = torch.where(lower_mask, 0, idx)
        elif m_lower == "warp":
            idx = torch.where(lower_mask, idx % shape, idx)
        elif m_lower == "drop":
            idx = torch.where(lower_mask, 0, idx)
            corner_volume = torch.where(lower_mask, 0.0, corner_volume)

        upper_mask = idx >= shape
        if m_upper == "clamp":
            idx = torch.where(upper_mask, shape - 1, idx)
        elif m_upper == "warp":
            idx = torch.where(upper_mask, idx % shape, idx)
        elif m_upper == "drop":
            idx = torch.where(upper_mask, shape - 1, idx)
            corner_volume = torch.where(upper_mask, 0.0, corner_volume)

        cube_corners_index_processed.append(idx)
    cube_corners_index = torch.stack(cube_corners_index_processed, -1)

    num_channels = points_weights.shape[-1] if points_weights is not None else 1
    features_flat = torch.zeros(batch_size, num_channels, math.prod(volume_shape), device=points.device)
    if points_weights is not None:
        feat_to_scatter = (
            corner_volume[:, :, :, None] * points_weights[:, :, None, :]
        )  # [B, N, 8, 1] * [B, N, 1, C] = [B, N, 8, C]

    else:
        feat_to_scatter = corner_volume[:, :, :, None]
    feat_to_scatter = einops.rearrange(feat_to_scatter, "b n m c -> b c (n m)")
    stride = torch.tensor(
        [volume_shape[1] * volume_shape[2], volume_shape[2], 1],
        device=cube_corners_float_index.device,
        dtype=torch.long,
    )
    cube_corners_index_flat = einops.rearrange(cube_corners_index, "b n m c -> b (n m) c")
    volume_index_flat = (stride * cube_corners_index_flat).sum(-1)  # [B, N * 8]

    features_flat.scatter_add_(2, volume_index_flat.unsqueeze(1).expand_as(feat_to_scatter), feat_to_scatter)
    features = features_flat.unflatten(-1, volume_shape)
    return features


def spherical_reflection_and_transmission_from_lidar(
    points_lidar: Tensor,
    points_weight: Tensor,
    spherical_volume: Volume,
    spherical_shape: Sequence[int],
    normalize: Optional[Literal["volume"]] = None,
):
    """Occupancy estimation from pointclouds."""

    points_lidar_spherical = cartesian_to_spherical(points_lidar)

    # add one bin along r
    spherical_shape_with_inf = [spherical_shape[0] + 1, spherical_shape[1], spherical_shape[2]]
    range_voxel_size = spherical_volume.voxel_size_from_shape(spherical_shape)[0, 0].item()

    spherical_volume_with_inf = Volume(
        lower=spherical_volume.lower,
        upper=spherical_volume.upper + torch.tensor([range_voxel_size, 0.0, 0.0], device=spherical_volume.upper.device),
    )

    reflection_and_transmission = volume_density_from_points(
        volume_shape=spherical_shape_with_inf,
        volume=spherical_volume_with_inf,
        points=points_lidar_spherical,
        points_weights=points_weight,
        modes=(("drop", "clamp"), "clamp", "warp"),  # clamp upper range, drop lower range
    )
    num_reflections = reflection_and_transmission

    # exclusive cumulative sum from end to start equal to
    # num_transmissions = (
    #     reflection_and_transmission.flip(2).cumsum(2).flip(2) - reflection_and_transmission
    # )
    num_transmissions = reflection_and_transmission.sum(2, keepdim=True) - reflection_and_transmission.cumsum(2)
    # drop infinity bin
    reflection_and_transmission = torch.cat(
        [num_reflections[:, :, :-1, :, :], num_transmissions[:, :, :-1, :, :]], 1
    )  # [B, C, R, PHI, THETA]
    if normalize == "volume":
        grid = spherical_volume.coord_grid(reflection_and_transmission, expand=False)
        voxel_size = spherical_volume.voxel_size(reflection_and_transmission)
        elementwise_volume = spherical_volume_element(
            grid - 0.5 * voxel_size[:, None, None, None, :], grid + 0.5 * voxel_size[:, None, None, None, :]
        )

        reflection_and_transmission = reflection_and_transmission / elementwise_volume.unsqueeze(1)

    return reflection_and_transmission


def occupancy_from_points(
    points_lidar: Tensor,
    points_weight: Tensor,
    lidar_from_ego: Tensor,
    cartesian_volume: Volume,
    spherical_volume: Volume,
    cartesian_shape: Sequence[int],
    spherical_shape: Sequence[int],
):
    """Occupancy estimation from pointclouds."""

    points_lidar_spherical = cartesian_to_spherical(points_lidar)

    density_spherical = volume_density_from_points(
        volume_shape=spherical_shape,
        volume=spherical_volume,
        points=points_lidar_spherical,
        points_weights=points_weight,
        modes=(("drop", "clamp"), "clamp", "warp"),  # clamp upper range, drop lower range
    )
    num_reflections = density_spherical
    num_transmissions = density_spherical.flip(2).cumsum(2).flip(2) - density_spherical  # exclusive cumsum
    density_spherical = torch.cat([num_reflections, num_transmissions], 1)

    grid_ego = cartesian_volume.new_coord_grid(cartesian_shape)
    grid_lidar = einsum_transform("ble,bxyze->bxyzl", lidar_from_ego, points=grid_ego)
    grid_lidar_spherical = cartesian_to_spherical(grid_lidar)
    density_sampled = spherical_volume.sample_volume(density_spherical, grid_lidar_spherical)

    scale = cartesian_volume.voxel_size_from_shape(cartesian_shape).prod(-1) / spherical_volume_element(
        grid_lidar_spherical - 0.5 * spherical_volume.voxel_size_from_shape(spherical_shape)[:, None, None, None, :],
        grid_lidar_spherical + 0.5 * spherical_volume.voxel_size_from_shape(spherical_shape)[:, None, None, None, :],
    ).clamp_min(1e-6)
    density_sampled = density_sampled * scale[:, None]
    return density_sampled
