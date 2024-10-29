"""Polars NuScenes parsing."""

from collections.abc import Sequence
from typing import Literal, Optional

import polars as pl


def token(input_name, output_name):
    """Token column."""
    return pl.when(pl.col(input_name) != "").then(pl.col(input_name)).otherwise(None).alias(output_name)


def rename(input_name, output_name, dtype: Optional[pl.DataType] = None):
    """Rename and cast.."""
    col = pl.col(input_name).alias(output_name)
    if dtype is not None:
        col = col.cast(dtype)
    return col


def array(input_name, output_name, shape: Sequence[int], dtype=pl.Float32):
    """Rename and cast to array."""
    array_type = dtype
    for dim in shape[::-1]:
        array_type = pl.Array(width=dim, inner=array_type)
    return pl.col(input_name).alias(output_name).cast(array_type)


def timestamp(input_name, output_name):
    """Rename and cast to timestamp."""
    return rename(input_name, output_name, dtype=pl.Datetime("us"))


def camera_intrinsic(input_name, output_name):
    """Rename and cast to camera intrinsic."""
    return (
        pl.when(pl.col(input_name).list.len() > 0)
        .then(pl.col(input_name))
        .otherwise(None)
        # identity matrix if no intrinsics are given to allow cast to array
        .fill_null([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .cast(pl.Array(width=3, inner=pl.Array(width=3, inner=pl.Float32)))
        .alias(output_name)
    )


def attribute_token(input_name, output_name):
    """Rename and cast to attribute token."""
    return pl.col(input_name).list.first().alias(output_name)


ATTRIBUTE_SELECTION = (
    token("token", "attribute.token"),
    rename("name", "attribute.name"),
    rename("description", "attribute.description"),
)


CALIBRATED_SENSOR_SELECTION = (
    token("token", "calibrated_sensor.token"),
    token("sensor_token", "sensor.token"),
    array("translation", "calibrated_sensor.translation", [3]),
    array("rotation", "calibrated_sensor.rotation", [4]),
    camera_intrinsic("camera_intrinsic", "calibrated_sensor.camera_intrinsic"),
)


CATEGORY_SELECTION = (
    token("token", "category.token"),
    rename("name", "category.name"),
    rename("description", "category.description"),
    rename("index", "category.index"),
)

EGO_POSE_SELECTION = (
    token("token", "ego_pose.token"),
    array("translation", "ego_pose.translation", [3]),
    array("rotation", "ego_pose.rotation", [4]),
    timestamp("timestamp", "ego_pose.timestamp"),
)

INSTANCE_SELECTION = (
    token("token", "instance.token"),
    token("category_token", "category.token"),
    token("first_annotation_token", "first_annotation.token"),
    token("last_annotation_token", "last_annotation.token"),
    rename("nbr_annotations", "instance.nbr_annotations"),
)

LIDARSEG_SELECTION = (
    token("token", "LIDAR_TOP.lidarseg.token"),
    token("sample_data_token", "LIDAR_TOP.sample_data.token"),
    rename("filename", "LIDAR_TOP.lidarseg.filename"),
)


LOG_SELECTION = (
    token("token", "log.token"),
    rename("logfile", "log.logfile"),
    rename("vehicle", "log.vehicle"),
    rename("date_captured", "log.date_captured"),
    rename("location", "log.location"),
)


MAP_SELECTION = (
    token("token", "map.token"),
    rename("category", "map.category"),
    rename("filename", "map.filename"),
    rename("log_tokens", "map.log_tokens"),
)


SAMPLE_SELECTION = (
    token("token", "sample.token"),
    token("next", "sample.next"),
    token("prev", "sample.prev"),
    token("scene_token", "scene.token"),
    timestamp("timestamp", "sample.timestamp"),
)


SAMPLE_ANNOTATION_SELECTION = (
    token("token", "sample_annotation.token"),
    token("next", "sample_annotation.next"),
    token("prev", "sample_annotation.prev_token"),
    token("sample_token", "sample.token"),
    token("instance_token", "instance.token"),
    token("visibility_token", "visibility.token"),
    attribute_token(
        "attribute_tokens", "attribute.token"
    ),  # there is always ONE attribute in this list -> single token
    array("translation", "sample_annotation.translation", [3]),
    array("rotation", "sample_annotation.rotation", [4]),
    array("size", "sample_annotation.size", [3]),
    rename("num_lidar_pts", "sample_annotation.num_lidar_pts"),
    rename("num_radar_pts", "sample_annotation.num_radar_pts"),
)

SAMPLE_DATA_SELECTION = (
    token("token", "sample_data.token"),
    token("next", "sample_data.next"),
    token("prev", "sample_data.prev"),
    token("sample_token", "sample.token"),
    token("ego_pose_token", "ego_pose.token"),
    token("calibrated_sensor_token", "calibrated_sensor.token"),
    rename("filename", "sample_data.filename"),
    rename("fileformat", "sample_data.fileformat"),
    rename("width", "sample_data.width"),
    rename("height", "sample_data.height"),
    timestamp("timestamp", "sample_data.timestamp"),
    rename("is_key_frame", "sample_data.is_key_frame"),
)

SCENE_SELECTION = (
    token("token", "scene.token"),
    token("log_token", "log.token"),
    token("first_sample_token", "first_sample.token"),
    token("last_sample_token", "last_sample.token"),
    rename("name", "scene.name"),
    rename("description", "scene.description"),
    rename("nbr_samples", "scene.nbr_samples"),
)

SENSOR_SELECTION = (
    token("token", "sensor.token"),
    rename("channel", "sensor.channel"),
    rename("modality", "sensor.modality"),
)


VISIBILITY_SELECTION = (
    token("token", "visibility.token"),
    rename("level", "visibility.level"),
    rename("description", "visibility.description"),
)
PANOPTIC_SELECTION = (
    token("token", "LIDAR_TOP.panoptic.token"),
    token("sample_data_token", "LIDAR_TOP.sample_data.token"),
    rename("filename", "LIDAR_TOP.panoptic.filename"),
)


ANNOTATION_FILES = Literal[
    "log",
    "scene",
    "instance",
    "category",
    "map",
    "sample",
    "lidarseg",
    "calibrated_sensor",
    "sample_data",
    "sample_annotation",
    "attribute",
    "sensor",
    "ego_pose",
    "visibility",
    "panoptic",
]

SELECTION: dict[ANNOTATION_FILES, Sequence[pl.Expr]] = {
    "log": LOG_SELECTION,
    "scene": SCENE_SELECTION,
    "instance": INSTANCE_SELECTION,
    "category": CATEGORY_SELECTION,
    "map": MAP_SELECTION,
    "sample": SAMPLE_SELECTION,
    "lidarseg": LIDARSEG_SELECTION,
    "calibrated_sensor": CALIBRATED_SENSOR_SELECTION,
    "sample_data": SAMPLE_DATA_SELECTION,
    "sample_annotation": SAMPLE_ANNOTATION_SELECTION,
    "attribute": ATTRIBUTE_SELECTION,
    "sensor": SENSOR_SELECTION,
    "ego_pose": EGO_POSE_SELECTION,
    "visibility": VISIBILITY_SELECTION,
    "panoptic": PANOPTIC_SELECTION,
}
