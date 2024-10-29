"""Plotly volumes."""

import plotly.graph_objects as go
from torch import Tensor

from scene_reconstruction.core.volume import Volume

from ..utils.mesh import cube_mesh_colored


def plotly_mesh_from_volume(
    features: Tensor,
    volume: Volume,
    batch_index: int = 0,
    channel_index: int = 0,
    colormap="turbo",
    scale: tuple[float, float] = (0.0, 1.0),
):
    """Generate plotly mesh from volume."""
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
    vertices_flat, triangles_flat, vertex_color = cube_mesh_colored(
        cube_centers=centers_flat, cube_sizes=voxel_size, cube_colors=features_flat[:, None]
    )
    upper = volume.upper.expand(features.shape[0], -1)
    mesh = go.Mesh3d(
        # vertices
        x=vertices_flat[:, 0].numpy(),
        y=vertices_flat[:, 1].numpy(),
        z=vertices_flat[:, 2].numpy(),
        # triangles
        i=triangles_flat[:, 0].numpy(),
        j=triangles_flat[:, 1].numpy(),
        k=triangles_flat[:, 2].numpy(),
        # color
        cmin=0.0,
        cmax=1.0,
        # vertexcolor=vertex_color[:, 0].numpy()
        intensity=vertex_color[:, 0].numpy(),
        colorscale=colormap,
        # lightning
        flatshading=True,
        lighting={
            "ambient": 0.4,
            "diffuse": 1,
            "fresnel": 0.1,
            "specular": 0.1,
            "roughness": 0.9,
            "vertexnormalsepsilon": 0,
            "facenormalsepsilon": 0,
        },
        lightposition={
            "x": upper[batch_index, 0].item(),
            "y": upper[batch_index, 1].item(),
            "z": upper[batch_index, 2].item(),
        },
        #
        hoverinfo="skip",
    )
    return mesh


def plotly_voxel_volume(
    features: Tensor,
    volume: Volume,
    batch_index: int = 0,
    channel_index: int = 0,
    colormap="turbo",
    scale: tuple[float, float] = (0.0, 1.0),
    show_grid: bool = False,
    equal_axis: bool = True,
):
    """Plot volumes using plotly."""
    fig = go.Figure(
        data=[
            plotly_mesh_from_volume(
                features,
                volume,
                batch_index=batch_index,
                channel_index=channel_index,
                colormap=colormap,
                scale=scale,
            )
        ],
        layout={
            "scene": {
                "aspectmode": "data" if equal_axis else "auto",
                "xaxis": {"visible": show_grid},
                "yaxis": {"visible": show_grid},
                "zaxis": {"visible": show_grid},
            }
        },
    )
    return fig
