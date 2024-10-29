"""Data input / output from files."""

from typing import Optional

import numpy as np
import torch


def load_lidar_from_file(filename: str, pad_length: Optional[int] = None):
    """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index)."""
    dims_to_load = [0, 1, 2]  # [x, y, z]
    scan = np.fromfile(filename, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, dims_to_load]
    if pad_length is None:
        return points
    points_padded = np.empty((pad_length, 3), dtype=points.dtype)
    num_points = len(points)
    points_padded[:num_points] = points
    points_padded[num_points:] = float("nan")
    return points_padded


def load_occupancy_from_file(filename: str) -> dict[str, torch.Tensor]:
    """Loads occupancy data from file."""
    data = np.load(file=filename)
    data = {k: torch.as_tensor(v) for k, v in data.items()}
    return data
