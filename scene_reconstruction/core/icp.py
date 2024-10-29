"""ICP using open3d."""

import open3d as o3d
import torch
from torch import Tensor


def torch_to_pcl(points: Tensor):
    """Torch  tensor of shape [N, 3] to open3d pointcloud."""
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    return pcl


def register_frame(ref_points: Tensor, new_points: Tensor):
    """ICP registration using open3d."""
    ref_from_guess = torch.as_tensor(
        o3d.pipelines.registration.registration_icp(
            torch_to_pcl(new_points),
            torch_to_pcl(ref_points),
            max_correspondence_distance=0.2,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        ).transformation,
        dtype=torch.float,
    )
    return ref_from_guess.to(device=ref_points.device)
