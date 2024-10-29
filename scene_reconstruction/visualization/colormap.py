"""Colormaps."""

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

occupancy_to_color = {
    "ignore_class": (0, 0, 0),  # Black.
    "barrier": (112, 128, 144),  # Slategrey
    "bicycle": (220, 20, 60),  # Crimson
    "bus": (255, 127, 80),  # Coral
    "car": (255, 158, 0),  # Orange
    "construction_vehicle": (233, 150, 70),  # Darksalmon
    "motorcycle": (255, 61, 99),  # Red
    "pedestrian": (0, 0, 230),  # Blue
    "traffic_cone": (47, 79, 79),  # Darkslategrey
    "trailer": (255, 140, 0),  # Darkorange
    "truck": (255, 99, 71),  # Tomato
    "driveable_surface": (0, 207, 191),  # nuTonomy green
    "other_flat": (175, 0, 75),
    "sidewalk": (75, 0, 75),
    "terrain": (112, 180, 60),
    "manmade": (222, 184, 135),  # Burlywood
    "vegetation": (0, 175, 0),
    "free": (52, 107, 235),
}
occupancy_color_map = torch.tensor(list(occupancy_to_color.values()))


def _create_turbo_alpha():
    newcolors = colormaps["turbo"](np.linspace(0, 1, 256))
    newcolors[0, -1] = 0.0
    turbo_alpha = ListedColormap(newcolors)
    return turbo_alpha


turbo_alpha = _create_turbo_alpha()


def _create_turbo_black():
    newcolors = colormaps["turbo"](np.linspace(0, 1, 256))
    newcolors[0, :3] = 0.0
    turbo_alpha = ListedColormap(newcolors)
    return turbo_alpha


turbo_black = _create_turbo_black()
