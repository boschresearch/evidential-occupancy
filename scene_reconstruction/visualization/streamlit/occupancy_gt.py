"""Visualize occupancy."""

import streamlit as st

from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.io import load_occupancy_from_file
from scene_reconstruction.visualization.plotly.volume import plotly_voxel_volume

st.set_page_config(layout="wide")

st.sidebar.title("Occupancy")

file = st.sidebar.file_uploader("Choose a file", type="npz")

use_camera_mask = st.sidebar.toggle("Camera mask")
use_lidar_mask = st.sidebar.toggle("Lidar mask")
show_free = st.sidebar.toggle("Show free")
hide_visible_freespace = st.sidebar.toggle("Hide visible free")
if file is not None:
    occ_data = load_occupancy_from_file(file)
    lower = [-40, -40.0, -1.0]
    upper = [40.0, 40.0, 5.4]
    volume = Volume.new_volume(lower, upper)
    semantics = occ_data["semantics"][None, None]
    mask_camera = occ_data["mask_camera"][None, None].bool()
    mask_lidar = occ_data["mask_lidar"][None, None].bool()
    if not show_free:
        semantics[semantics == 17] = -1
    if use_camera_mask:
        semantics[~mask_camera] = -1
    if use_lidar_mask:
        semantics[~mask_lidar] = -1
    if hide_visible_freespace:
        semantics[mask_camera & (semantics == 17)] = -1
    fig = plotly_voxel_volume(semantics, volume, colormap="turbo_r", scale=(0, 17))
    fig.update_layout(height=900)
    st.plotly_chart(fig, use_container_width=True)
