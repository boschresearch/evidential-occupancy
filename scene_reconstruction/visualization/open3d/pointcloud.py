"""Open3d pointcloud visualization."""

from typing import Optional

import numpy as np
import open3d as o3d
from torch import Tensor


def open3d_pcl_from_torch(points: Tensor, color: Optional[Tensor] = None):
    """Colored open3d pointcloud form torch."""
    vec = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcl = o3d.geometry.PointCloud(vec)
    if color is not None:
        pcl.colors = o3d.utility.Vector3dVector(color.cpu().numpy())
    return pcl


def show_pointcloud(points: np.ndarray, color: Optional[np.ndarray] = None):
    """Colored pointcloud visualization."""
    vec = o3d.utility.Vector3dVector(points)
    pcl = o3d.geometry.PointCloud(vec)
    if color is not None:
        pcl.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([pcl])  # pylint: disable=E1101
