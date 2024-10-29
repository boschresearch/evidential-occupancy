"""Mesh."""

import torch
from torch import Tensor

CUBE_CORNERS = (
    torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float,
    )
    - 0.5
)  # [8, 3]
NUM_CUBE_CORNERS = len(CUBE_CORNERS)  # 8
CUBE_TRIANGLES = torch.tensor(
    [
        [4, 7, 5],
        [4, 6, 7],
        [0, 2, 4],
        [2, 6, 4],
        [0, 1, 2],
        [1, 3, 2],
        [1, 5, 7],
        [1, 7, 3],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 1],
        [1, 4, 5],
    ],
    dtype=torch.int32,
)  # [12, 3]
CUBE_TRIANGLE_NORMALS = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=torch.float,
)  # [12, 3]
NUM_CUBE_TRIANGLES = len(CUBE_TRIANGLES)  # 12


def cube_mesh(
    cube_centers: Tensor,
    cube_sizes: Tensor,
):
    """Generate vertices and triangles from cube centers and sizes."""
    # cube_centers: [N, 3]
    # cube_size: [N, 3], [N, 1]
    assert cube_centers.ndim == 2 and cube_centers.shape[1] == 3
    cube_sizes = cube_sizes.expand_as(cube_centers)
    num_cubes = cube_centers.shape[0]
    cube_corners = CUBE_CORNERS.clone()
    if cube_sizes is not None:
        cube_corners = cube_sizes[:, None, :] * cube_corners
    vertices = cube_centers[:, None, :] + cube_corners  # [N, CUBE_CORNERS, 3]

    triangles = NUM_CUBE_CORNERS * torch.arange(num_cubes)[:, None, None] + CUBE_TRIANGLES  # [N, CUBE_TRIANGLES, 3]

    vertices_flat = vertices.reshape((-1, 3))  # [N * CUBE_CORNERS, 3]
    triangles_flat = triangles.reshape((-1, 3))  # [N * NUM_CUBE_CORNERS, 3]

    return vertices_flat, triangles_flat


def cube_mesh_colored(
    cube_centers: Tensor,
    cube_sizes: Tensor,
    cube_colors: Tensor,
):
    """Generate vertices, triangles and vertex colors from cube centers, sizes and colors."""
    # cube_centers: [N, 3]
    # cube_size: [N, 3], [N, 1]
    # cube_colors: [N, 3], [N, 1]
    vertices_flat, triangles_flat = cube_mesh(cube_centers, cube_sizes)
    color_channels = cube_colors.shape[-1]
    vertex_colors = cube_colors[:, None, :].expand(-1, NUM_CUBE_CORNERS, -1).reshape((-1, color_channels))
    assert vertices_flat.shape[0] == vertex_colors.shape[0]
    return vertices_flat, triangles_flat, vertex_colors
