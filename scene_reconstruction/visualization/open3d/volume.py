"""Open3d dense grids."""

import matplotlib as mpl
import numpy as np
import open3d as o3d
from torch import Tensor

from scene_reconstruction.core import Volume

from ..utils.mesh import cube_mesh_colored


def open3d_mesh_from_volume(
    features: Tensor,
    volume: Volume,
    batch_index: int = 0,
    channel_index: int = 0,
    colormap="turbo",
    scale: tuple[float, float] = (0.0, 1.0),
    legacy: bool = True,
):
    """Generate open3d mesh from volume."""
    centers = volume.coord_grid(features)[batch_index]  # [X, Y, Z, 3]
    features = features[batch_index, channel_index]  # [X, Y, Z]
    if features.is_floating_point():
        mask = ~features.isnan()
    else:
        mask = features != -1
    features = (features - scale[0]) / (scale[1] - scale[0])
    centers_flat = centers[mask]  # [N, 3]
    features_flat = features[mask]
    voxel_size = volume.voxel_size(features)
    if voxel_size.shape[0] != 1:
        voxel_size = voxel_size[batch_index]
    vertices_flat, triangles_flat, vertex_color = cube_mesh_colored(
        cube_centers=centers_flat, cube_sizes=voxel_size, cube_colors=features_flat[:, None]
    )
    vertex_colors = mpl.colormaps[colormap](vertex_color.numpy()[:, 0])[:, :3].astype(np.float32)
    if legacy:
        vertices = o3d.utility.Vector3dVector(vertices_flat.numpy())
        triangles = o3d.utility.Vector3iVector(triangles_flat.numpy())

        colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        mesh.vertex_colors = colors
        mesh.compute_triangle_normals()

    else:
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(vertices_flat.numpy())
        mesh.triangle.indices = o3d.core.Tensor(triangles_flat.numpy())
        mesh.vertex.colors = o3d.core.Tensor(vertex_colors)
        mesh.compute_triangle_normals()
        mesh.material = o3d.visualization.Material("defaultLit")
        # mesh.material.scalar_properties["roughness"] = 0.8
        # mesh.material.scalar_properties["reflectance"] = 0.3
        # mesh.material.scalar_properties["transmission"] = 0.0
        # mesh.material.scalar_properties["thickness"] = 0.3
        # mesh.material.scalar_properties["absorption_distance"] = 0.1
        # mesh.material.vector_properties['absorption_color'] = np.array([1.0, 1.0, 1.0, 1.0])
    return mesh


def open3d_voxel_volume(
    features: Tensor,
    volume: Volume,
    batch_index: int = 0,
    channel_index: int = 0,
    colormap="turbo",
    scale: tuple[float, float] = (0.0, 1.0),
    legacy: bool = True,
):
    """Plot volumes using open3d."""
    mesh = open3d_mesh_from_volume(
        features,
        volume,
        batch_index,
        channel_index,
        colormap,
        scale,
        legacy=legacy,
    )
    if legacy:
        o3d.visualization.draw_geometries([mesh])  # pylint: disable=E1101
    else:
        geometries = [{"name": "volume", "geometry": mesh}]
        o3d.visualization.draw(geometries, up=[0.0, 0.0, 1.0])
