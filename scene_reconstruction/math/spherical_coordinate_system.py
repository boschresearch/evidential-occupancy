"""Math related to spherical coordinates."""

import torch
from torch import Tensor

# https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions


def cartesian_to_spherical(xyz: Tensor):
    """Transforms cartestian coordinates into spherical coordinates."""
    # pylint: disable=C0103
    x, y, z = xyz.unbind(-1)
    r = xyz.norm(dim=-1)  # radius >= 0
    theta = torch.arccos(z / r)  # 0 <= polar angle <= pi
    phi = torch.atan2(y, x)  # -pi <= azimuth <= pi
    spherical_coords = torch.stack([r, theta, phi], -1)
    return spherical_coords


def spherical_to_cartesian(spherical_coords: Tensor):
    """Transforms spherical coordinates into cartestian coordinates."""
    # pylint: disable=C0103
    r, theta, phi = spherical_coords.unbind(-1)
    sin_theta = theta.sin()
    cos_theta = theta.cos()
    sin_phi = phi.sin()
    cos_phi = phi.cos()
    r_sin_theta = r * sin_theta
    x = r_sin_theta * cos_phi
    y = r_sin_theta * sin_phi
    z = r * cos_theta
    xyz = torch.stack([x, y, z], -1)
    return xyz


def spherical_volume_element(spherical_lower: Tensor, spherical_upper: Tensor):
    """Calculates to volume of a spherical volume element specified by its lower and upper bound."""
    r_lower, theta_lower, phi_lower = spherical_lower.unbind(-1)
    r_upper, theta_upper, phi_upper = spherical_upper.unbind(-1)
    # https://en.wikipedia.org/wiki/Multiple_integral#Spherical_coordinates
    volume = (phi_upper - phi_lower) * (-theta_upper.cos() + theta_lower.cos()) * (r_upper**3 - r_lower**3) / 3.0
    return volume


def spherical_volume_element_center_and_voxel_size(spherical_center: Tensor, spherical_voxel_size: Tensor):
    """Calculates to volume of a spherical volume element specified by its center and voxel size."""
    r, theta, phi = spherical_center.unbind(-1)
    dr, dtheta, dphi = spherical_voxel_size.unbind(-1)

    # https://en.wikipedia.org/wiki/Multiple_integral#Spherical_coordinates
    volume = (
        dphi
        * (-(theta + 0.5 * dtheta).cos() + (theta - 0.5 * dtheta).cos())
        * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)
        / 3.0
    )
    return volume
