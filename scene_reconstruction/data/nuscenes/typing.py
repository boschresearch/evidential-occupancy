"""NuScenes typing."""

from typing import Literal, Union

# pylint: disable=C0103

CAMERA_CHANNELS = Literal[
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
LIDAR_CHANNELS = Literal["LIDAR_TOP"]
RADAR_CHANNELS = Literal[
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]
SENSOR_CHANNELS = Union[CAMERA_CHANNELS, LIDAR_CHANNELS, RADAR_CHANNELS]

SENSOR_MODALITY = Literal["camera", "lidar", "radar"]
CATEGOTY_NAMES = Literal[
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
]
