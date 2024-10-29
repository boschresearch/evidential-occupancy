"""Homogenoues transformation related functions."""

import logging

import torch
from torch import Tensor


def transform_volume_bounds(
    lower: Tensor,
    upper: Tensor,
    new_lower: Tensor,
    new_upper: Tensor,
    verbose: bool = False,
):
    """Homogenous transformation matrix between volumes specified by lower and upper bounds."""
    # xyz = ((xyz - lower) / (upper - lower)) * (new_upper - new_lower) + new_lower
    scale = (new_upper - new_lower) / (upper - lower)

    if verbose and not scale.allclose(scale[..., 0:1]):
        logging.getLogger(__file__).warning(f"Scale is not equal across all axis: {scale}")
    scale_homogenous = torch.nn.functional.pad(scale, (0, 1), "constant", 1.0)  # pylint: disable=E1102
    offset = new_lower - lower * scale
    transform = torch.diag_embed(scale_homogenous)
    transform[..., :3, 3] = offset
    return transform


def to_homogenous(xyz: Tensor):
    """Converts coordinates into homogenous coordinates."""
    return torch.nn.functional.pad(xyz, (0, 1), "constant", 1.0)  # pylint: disable=E1102


def from_homogenous(xyzh: Tensor):
    """Converts homogenous coordinates into normal coordinates."""
    return xyzh[..., :3]


def transform_to_grid_sample_coords(lower, upper):
    """Converts from bounded volume to coordintates which can be used with grid_sample."""
    # Grid sample uses normalized coordinates between -1.0 and 1.0
    # Addionally xyz needs to be reversed to zyx
    normalization_transform = transform_volume_bounds(lower, upper, -torch.ones_like(lower), torch.ones_like(upper))
    swap_transform = torch.eye(4, device=lower.device, dtype=lower.dtype)[[2, 1, 0, 3]]
    transform = swap_transform @ normalization_transform
    return transform


def einsum_transform(einsum_str: str, *transform: Tensor, points: Tensor):
    """Applies a transformation to points using the einsum notation."""
    is_homogenous = points.shape[-1] == 4
    if not is_homogenous:
        points = to_homogenous(points)
        assert points.shape[-1] == 4
    points = torch.einsum(einsum_str, *transform, points)
    if not is_homogenous:
        points = from_homogenous(points)
    return points
