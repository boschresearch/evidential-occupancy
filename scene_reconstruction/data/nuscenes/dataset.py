"""NuScenes Dataset."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Union, get_args

import numpy as np
import polars as pl
import tqdm
from PIL import Image
from polars.type_aliases import JoinStrategy, JoinValidation

from .parse import ANNOTATION_FILES, SELECTION
from .polars_helpers import (
    join_on_token,
    numpy_to_series,
    pad_intrinsics_from_colums,
    series_to_numpy,
    transform_from_columns,
)
from .typing import CAMERA_CHANNELS, LIDAR_CHANNELS, RADAR_CHANNELS, SENSOR_CHANNELS

# pylint: disable=C0103,R0902

CAMERAS = get_args(CAMERA_CHANNELS)
LIDARS = get_args(LIDAR_CHANNELS)
ALL_CHANNELS = get_args(SENSOR_CHANNELS)

VERSION = Literal["v1.0-mini", "v1.0-trainval"]


class NuscenesDataset:
    """Nuscenes Dataset."""

    def __init__(
        self,
        data_root: Union[Path, str] = "./data/nuscenes/",
        extra_data_root: Optional[Union[Path, str]] = None,
        version: VERSION = "v1.0-mini",
        verbose: bool = False,
        key_frames_only: bool = True,
        with_sample_annotation: bool = False,
        with_lidarseg: bool = False,
        with_panoptic: bool = False,
    ) -> None:
        """Initializes NuScenes dataset from data root."""
        self.data_root = Path(data_root)
        self.extra_data_root = Path(extra_data_root) if extra_data_root is not None else None
        self.version: VERSION = version
        self.key_frames_only = key_frames_only

        # load annotation files
        # annotations schema: https://www.nuscenes.org/nuscenes#data-annotation
        self.log = self.read_json("log")
        self.scene = self.read_json("scene")
        self.instance = self.read_json("instance")
        self.category = self.read_json("category")
        self.map = self.read_json("map")
        self.sample = self.read_json("sample")
        self.calibrated_sensor = self.read_json("calibrated_sensor")
        self.sample_data = self.read_json("sample_data")
        self.sample_annotation = self.read_json("sample_annotation")
        self.attribute = self.read_json("attribute")
        self.sensor = self.read_json("sensor")
        self.ego_pose = self.read_json("ego_pose")
        self.visibility = self.read_json("visibility")
        if with_lidarseg:
            self.lidarseg = self.read_json("lidarseg")
        if with_panoptic:
            self.panoptic = self.read_json("panoptic")
        self.sample_data_dict: dict[SENSOR_CHANNELS, pl.DataFrame]
        self.sample_annotation_dict: dict[SENSOR_CHANNELS, pl.DataFrame] = {}

        self.lidar_pad_length = 35_000
        self.verbose = verbose
        self.with_sample_annotation = with_sample_annotation

        self._prepare(key_frames_only)

    def read_json(self, name: ANNOTATION_FILES):
        """Reads NuScenes json files."""
        return pl.read_json(self.data_root / self.version / f"{name}.json").select(SELECTION[name])

    def _assign_sample_keyframe_index(self, sample: pl.DataFrame):
        sample = sample.sort("scene.token", "sample.timestamp").with_columns(
            pl.arange(0, pl.count()).over("scene.token").alias("sample.key_frame_index")
        )
        return sample

    @staticmethod
    def _box_to_dict(sample_data_token_name: str, sample_data_token_name_value: str, box):
        return {
            sample_data_token_name: sample_data_token_name_value,
            "sample_annotation.token": box.token,
            "sample_annotation.translation": list(box.center),
            "sample_annotation.size": list(box.wlh),
            "sample_annotation.rotation": list(box.orientation.q),
        }

    def load_sensor_sample_annotation(self, sensor_channel: SENSOR_CHANNELS):
        """Loads sensor synced sample annotation from cache or builds it from scratch."""
        if self.extra_data_root is not None:
            filename = self.extra_data_root / "sample_annotation_cache" / f"{sensor_channel}.arrow"
            if filename.exists():
                self.sample_annotation_dict[sensor_channel] = pl.read_ipc(filename, memory_map=False)
                return
        self.build_sensor_sample_annotation(sensor_channel)
        if self.extra_data_root is not None:
            filename.parent.mkdir(exist_ok=True, parents=True)
            self.sample_annotation_dict[sensor_channel].write_ipc(filename, compression="zstd")

    def build_sensor_sample_annotation(self, sensor_channel: SENSOR_CHANNELS):
        """Interpolates sample annotation to given sample data token."""
        import nuscenes

        nuscenes = nuscenes.NuScenes(verbose=False, version=self.version, dataroot=self.data_root)

        sample_data_token_name = f"{sensor_channel}.sample_data.token"
        sample_data_tokens = self.sample_data_dict[sensor_channel][sample_data_token_name]

        boxes = []
        for sample_data_token in tqdm.tqdm(
            sample_data_tokens,
            desc=f"Build sample annotation for sensor {sensor_channel}",
        ):
            boxes.extend(
                [
                    self._box_to_dict(sample_data_token_name, sample_data_token, box)
                    for box in nuscenes.get_boxes(sample_data_token)
                ]
            )
        interpolated_annotations = pl.DataFrame(boxes).with_columns(
            pl.col(sample_data_token_name),
            pl.col("sample_annotation.token"),
            pl.col("sample_annotation.translation").cast(pl.Array(width=3, inner=pl.Float32)),
            pl.col("sample_annotation.size").cast(pl.Array(width=3, inner=pl.Float32)),
            pl.col("sample_annotation.rotation").cast(pl.Array(width=4, inner=pl.Float32)),
        )
        # join additional information
        interpolated_annotations = self.join(
            interpolated_annotations,
            self.sample_annotation.select(
                pl.exclude(
                    "sample_annotation.translation",
                    "sample_annotation.size",
                    "sample_annotation.rotation",
                    "transform.global_from_obj",
                    "transform.obj_from_global",
                    "sample_annotation.size_aligned",
                )
            ),
        )
        interpolated_annotations = self.prepare_sample_annotation(interpolated_annotations)

        self.sample_annotation_dict[sensor_channel] = interpolated_annotations

    def load_cvpr2023_occupancy(self, scene: pl.DataFrame, root_path: Optional[Union[Path, str]] = None):
        """Loads occupancy ground truth from disk."""

        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_name, sample_token = x
            return root_path / "nuscenes_occ3d" / "gts" / scene_name / sample_token / "labels.npz"

        filenames = list(map(make_filename, scene.select("scene.name", "sample.token").iter_rows()))
        data_dict: dict[str, list[np.ndarray]] = {}
        for filename in filenames:
            for k, v in np.load(filename).items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)
        list_of_series = [numpy_to_series(f"sample.occ_gt.{k}", np.stack(v)) for k, v in data_dict.items()]
        scene = scene.with_columns(*list_of_series)
        scene = scene.sort("sample.timestamp")
        return scene

    def load_open_occpancy(self, scene: pl.DataFrame, root_path: Optional[Union[Path, str]] = None):
        """Loads occupancy ground truth from disk."""

        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_token, lidar_token = x
            return root_path / "nuScenes-Occupancy-v0.1" / f"scene_{scene_token}" / "occupancy" / f"{lidar_token}.npy"

        filenames = list(
            map(
                make_filename,
                scene.select("scene.token", "LIDAR_TOP.sample_data.token").iter_rows(),
            )
        )
        volume_shape = [512, 512, 40]
        num_samples = len(filenames)
        lower = np.broadcast_to(np.array([-51.2, -51.2, -5.0], dtype=np.float32), [len(filenames), 3])
        upper = np.broadcast_to(np.array([51.2, 51.2, 3.0], dtype=np.float32), [len(filenames), 3])
        dense_occupancy = np.full([num_samples] + volume_shape, fill_value=17, dtype=np.int32)
        for i, filename in enumerate(filenames):
            # https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md
            # https://github.com/JeffWang987/OpenOccupancy/blob/main/projects/occ_plugin/datasets/pipelines/loading.py
            #  [z y x cls]
            pcd = np.load(filename)
            pcd_label = pcd[..., -1]
            # map free and noise to 17
            pcd_label[pcd_label == 0] = 17
            pcd_label[pcd_label == 255] = 17
            z, y, x = pcd[:, 0], pcd[:, 1], pcd[:, 2]
            batch_idx = np.broadcast_to(np.array(i), x.shape)
            dense_occupancy[batch_idx, x, y, z] = pcd_label
        scene = scene.with_columns(
            numpy_to_series("sample.open_occupancy.semantics", dense_occupancy),
            numpy_to_series("sample.open_occupancy.volume.lower", lower),
            numpy_to_series("sample.open_occupancy.volume.upper", upper),
        )
        scene = scene.sort("sample.timestamp")
        return scene

    def load_surround_occ_occupancy(
        self,
        scene: pl.DataFrame,
        root_path: Optional[Union[Path, str]] = None,
        subdir: str = "surround_occ_occupancy",
    ):
        """Loads occupancy ground truth from disk."""

        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            token, pcd_filename = x
            filename = Path(pcd_filename).name
            return root_path / subdir / "samples" / f"{filename}.npy"

        filenames = list(
            map(
                make_filename,
                scene.select("scene.token", "LIDAR_TOP.sample_data.filename").iter_rows(),
            )
        )
        volume_shape = [200, 200, 16]
        num_samples = len(filenames)
        lower = np.broadcast_to(np.array([-50.0, -50.0, -5.0], dtype=np.float32), [len(filenames), 3])
        upper = np.broadcast_to(np.array([50.0, 50.0, 3.0], dtype=np.float32), [len(filenames), 3])
        dense_occupancy = np.full([num_samples] + volume_shape, fill_value=17, dtype=np.int32)
        for i, filename in enumerate(filenames):
            # https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md
            # https://github.com/weiyithu/SurroundOcc/blob/main/projects/mmdet3d_plugin/datasets/pipelines/loading.py
            # The ground truth is a (N, 4) tensor, N is the occupied voxel number,
            # The first three channels represent xyz voxel coordinate and last channel is semantic class.
            #  [z y x cls]
            pcd = np.load(filename)
            pcd_label = pcd[..., -1]
            # map free and noise to 17
            pcd_label[pcd_label == 0] = 17
            pcd_label[pcd_label == 255] = 17
            x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
            batch_idx = np.broadcast_to(np.array(i), x.shape)
            dense_occupancy[batch_idx, x, y, z] = pcd_label
        scene = scene.with_columns(
            numpy_to_series("sample.surround_occ.semantics", dense_occupancy),
            numpy_to_series("sample.surround_occ.volume.lower", lower),
            numpy_to_series("sample.surround_occ.volume.upper", upper),
        )
        scene = scene.sort("sample.timestamp")
        return scene

    def load_scene_as_occupancy_gt(
        self,
        scene: pl.DataFrame,
        root_path: Optional[Union[Path, str]] = None,
        subdir: str = "occ_gt_release_v1_0",
    ):
        """Loads occupancy ground truth from disk."""

        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_token, scene_name, key_frame_index = x
            return root_path / subdir / "trainval" / scene_name / f"{key_frame_index:03d}_occ.npy"

        filenames = list(
            map(
                make_filename,
                scene.select("scene.token", "scene.name", "sample.key_frame_index").iter_rows(),
            )
        )
        volume_shape = [200, 200, 16]
        num_voxels = math.prod(volume_shape)
        num_samples = len(filenames)
        lower = np.broadcast_to(np.array([-50.0, -50.0, -5.0], dtype=np.float32), [len(filenames), 3])
        upper = np.broadcast_to(np.array([50.0, 50.0, 3.0], dtype=np.float32), [len(filenames), 3])
        dense_occupancy = np.full([num_samples] + volume_shape, fill_value=17, dtype=np.int32)
        for i, filename in enumerate(filenames):
            # https://github.com/OpenDriveLab/OccNet/blob/main/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
            # https://github.com/OpenDriveLab/OccNet/blob/main/docs/prepare_dataset.md
            # The ground truth is a (N, 4) tensor, N is the occupied voxel number,
            # The first three channels represent xyz voxel coordinate and last channel is semantic class.
            #  [z y x cls]num_voxels = math.prod(volume_shape)
            data = np.load(filename)
            flat_zyx_index = data[:, 0]
            label = data[:, 1]
            dense_voxel_grid = np.full(num_voxels, fill_value=17, dtype=np.int32)  # [Z, Y, X]
            dense_voxel_grid[flat_zyx_index] = label
            # transpose to convert ZYX to XYZ
            dense_voxel_grid = dense_voxel_grid.reshape(volume_shape[::-1]).transpose(2, 1, 0)
            dense_occupancy[i] = dense_voxel_grid
        scene = scene.with_columns(
            numpy_to_series("sample.scene_as_occupancy.semantics", dense_occupancy),
            numpy_to_series("sample.scene_as_occupancy.volume.lower", lower),
            numpy_to_series("sample.scene_as_occupancy.volume.upper", upper),
        )
        scene = scene.sort("sample.timestamp")
        return scene

    def load_reflection_and_transmission_spherical(
        self, scene: pl.DataFrame, root_path: Optional[Union[Path, str]] = None
    ):
        """Loads number of transmissions and reflections in sperical coords."""
        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_name, lidar_token = x
            return (
                root_path / "reflection_and_transmission_spherical" / scene_name / "LIDAR_TOP" / f"{lidar_token}.arrow"
            )

        filenames = list(
            map(
                make_filename,
                scene.select("scene.name", "LIDAR_TOP.sample_data.token").iter_rows(),
            )
        )
        df = pl.scan_ipc(filenames, memory_map=False)
        scene = scene.lazy().join(df, on="LIDAR_TOP.sample_data.token", validate="1:1", how="left").collect()
        return scene

    def load_reflection_and_transmission_multi_frame(
        self, scene: pl.DataFrame, root_path: Optional[Union[Path, str]] = None
    ):
        """Loads number of transmissions and reflections in ego frame accumulated over multiple frames coords."""
        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_name, lidar_token = x
            return (
                root_path
                / "reflection_and_transmission_multi_frame"
                / scene_name
                / "LIDAR_TOP"
                / f"{lidar_token}.arrow"
            )

        filenames = list(
            map(
                make_filename,
                scene.select("scene.name", "LIDAR_TOP.sample_data.token").iter_rows(),
            )
        )
        df = pl.scan_ipc(filenames, memory_map=False)
        scene = scene.lazy().join(df, on="LIDAR_TOP.sample_data.token", validate="1:1", how="left").collect()
        return scene

    def load_scene_flow_polars(self, scene: pl.DataFrame, root_path: Optional[Union[Path, str]] = None):
        """Loads scene flow information."""
        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_name, lidar_token = x
            return root_path / "scene_flow" / scene_name / "LIDAR_TOP" / f"{lidar_token}.arrow"

        filenames = list(
            map(
                make_filename,
                scene.select("scene.name", "LIDAR_TOP.sample_data.token").iter_rows(),
            )
        )
        df = pl.scan_ipc(filenames, memory_map=False)
        scene = scene.lazy().join(df, on="LIDAR_TOP.sample_data.token", validate="1:1", how="left").collect()
        return scene

    def load_occupancy_preds(
        self,
        scene: pl.DataFrame,
        name: str,
        root_path: Optional[Union[Path, str]] = None,
    ):
        """Loads occupancy preds truth from disk."""

        root_path = Path(root_path) if root_path is not None else self.extra_data_root

        def make_filename(x):
            scene_name, sample_token = x
            return root_path / name / f"{sample_token}.npz"

        filenames = list(map(make_filename, scene.select("scene.name", "sample.token").iter_rows()))
        data_dict: dict[str, list[np.ndarray]] = {}
        for filename in filenames:
            for k, v in np.load(filename).items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)
        list_of_series = [numpy_to_series(f"sample.{name}.{k}", np.stack(v)) for k, v in data_dict.items()]
        scene = scene.with_columns(*list_of_series)
        scene = scene.sort("sample.timestamp")
        return scene

    def sort_by_time(self, scene: pl.DataFrame):
        """Sort scene by time."""
        return scene.sort("LIDAR_TOP.sample_data.timestamp")

    def save_scenes(self):
        """Save scenes to disk."""
        for scene_name in tqdm.tqdm(self.scene["scene.name"]):
            scene = self.load_scene(scene_name)
            scene_dir = self.data_root / "scenes"
            scene_dir.mkdir(exist_ok=True)
            scene_name = scene.item(0, "scene.name")
            scene.write_ipc(
                scene_dir / f"{scene_name}.arrow",
                compression="zstd",
            )

    def load_scene_from_file(self, scene_name: str):
        """Load scene from file."""
        return pl.read_ipc(
            self.data_root / "scenes" / f"{scene_name}.arrow",
            memory_map=False,
        )

    def build_transformations(self):
        """Build homogenous transformations."""
        # sensor / ego
        self.calibrated_sensor = transform_from_columns(
            self.calibrated_sensor,
            "calibrated_sensor.rotation",
            "calibrated_sensor.translation",
            transform="transform.ego_from_sensor",
            transform_inv="transform.sensor_from_ego",
        )
        # sensor intrinsics
        self.calibrated_sensor = pad_intrinsics_from_colums(
            self.calibrated_sensor,
            "calibrated_sensor.camera_intrinsic",
            "transform.image_from_sensor",
        )
        # ego / global
        self.ego_pose = transform_from_columns(
            self.ego_pose,
            "ego_pose.rotation",
            "ego_pose.translation",
            transform="transform.global_from_ego",
            transform_inv="transform.ego_from_global",
        )

    def build_sample_annotation_transforms(self, sample_annotation: pl.DataFrame):
        """Build sample annotation transforms."""
        sample_annotation = transform_from_columns(
            sample_annotation,
            "sample_annotation.rotation",
            "sample_annotation.translation",
            transform="transform.global_from_obj",
            transform_inv="transform.obj_from_global",
        )
        return sample_annotation

    def filter_key_frames_only(self):
        """Filter by key frame."""
        self.sample_data = self.sample_data.filter(pl.col("sample_data.is_key_frame"))

    def build_sample_data_dict(self):
        """Build sample data dict per sensor."""
        self.sample_data_dict: dict[SENSOR_CHANNELS, pl.DataFrame] = {
            channel: df.select(pl.all().name.prefix(f"{channel}."))
            for (channel,), df in self.sample_data.partition_by("sensor.channel", as_dict=True).items()
        }

    def load_sample_data(
        self,
        sample: pl.DataFrame,
        sensor_channel: SENSOR_CHANNELS,
        with_data: bool = True,
    ):
        """Load sample data."""
        sample = self.join(
            sample,
            self.sample_data_dict[sensor_channel].rename({f"{sensor_channel}.sample.token": "sample.token"}),
            # validate="1:1" if self.key_frames_only else "1:m",
        )

        if with_data:
            sample = self._load_sensor_data(sample, sensor_channel)
        sample = self.sort_by_time(sample)
        return sample

    def load_panoptic(
        self,
        df: pl.DataFrame,
        with_data: bool = False,
    ):
        """Load panoptic info."""
        df = self.join(df, self.lidarseg, validate="1:1")
        if with_data:
            df = self._load_panoptic_data(df)
        return df

    def _load_panoptic_data(
        self,
        df: pl.DataFrame,
    ):
        """Load panoptic data."""
        data = np.stack(list(map(self._load_lidarseg_from_file, df["LIDAR_TOP.panoptic.filename"])))
        df = df.with_columns(numpy_to_series("LIDAR_TOP.panoptic.data", data))
        return df

    def load_lidarseg(
        self,
        df: pl.DataFrame,
        with_data: bool = False,
    ):
        """Load lidar segmentation data."""
        df = self.join(df, self.lidarseg, validate="1:1")
        if with_data:
            df = self._load_lidarseg_data(df)
        return df

    def _load_lidarseg_data(
        self,
        df: pl.DataFrame,
    ):
        data = np.stack(
            df["LIDAR_TOP.lidarseg.filename"]
            .map_elements(self._load_lidarseg_from_file, return_dtype=pl.Object())
            .to_list()
        )
        df = df.with_columns(numpy_to_series("LIDAR_TOP.points_category_idx", data))
        return df

    def _load_sensor_data(
        self,
        df: pl.DataFrame,
        sensor_channel: SENSOR_CHANNELS,
    ):
        """Loads sensor data."""
        if sensor_channel in get_args(CAMERA_CHANNELS):
            load_fn = self._load_image_from_file
            prefix = f"{sensor_channel}.sample_data"
            filename = f"{sensor_channel}.sample_data.filename"
        elif sensor_channel in get_args(LIDAR_CHANNELS):
            load_fn = self._load_lidar_from_file
            prefix = f"{sensor_channel}.sample_data"
            filename = f"{sensor_channel}.sample_data.filename"
        elif sensor_channel in get_args(RADAR_CHANNELS):
            raise NotImplementedError("Loading radar data is not yet supported")
        else:
            raise ValueError(f"Unknown channel: {sensor_channel}")
        list_of_dicts = [load_fn(f) for f in df[filename]]
        dict_of_data = {f"{prefix}.{k}": np.stack([d[k] for d in list_of_dicts]) for k in list_of_dicts[0].keys()}
        return df.with_columns(*[numpy_to_series(k, v) for k, v in dict_of_data.items()])

    def join(
        self,
        df: pl.DataFrame,
        other: pl.DataFrame,
        how: JoinStrategy = "inner",
        *,
        allow_duplicates: bool = False,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
    ):
        """Join frames on common tokens."""
        return join_on_token(
            df,
            other,
            how=how,
            suffix=suffix,
            validate=validate,
            verbose=self.verbose,
            allow_duplicates=allow_duplicates,
        )

    def _load_image_from_file(self, filename: str):
        """Load single image from file."""
        return {"image": np.asarray(Image.open(self.data_root / filename))}

    def _load_lidar_from_file(self, filename: str):
        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index)."""
        dims_to_load = [0, 1, 2]  # [x, y, z]
        scan = np.fromfile(self.data_root / filename, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, dims_to_load]
        points_padded = np.empty((self.lidar_pad_length, 3), dtype=points.dtype)
        num_points = len(points)
        points_padded[:num_points] = points
        points_padded[num_points:] = float("nan")
        mask = np.empty((self.lidar_pad_length,), dtype=bool)
        mask[:num_points] = 1
        mask[num_points:] = 0
        return {"points_lidar": points_padded, "points_mask": mask}

    def _load_lidarseg_from_file(self, filename: str):
        """Loads lidar seg data from file."""
        data = np.fromfile(self.data_root / filename, dtype=np.uint8)
        data_padded = np.empty(self.lidar_pad_length, dtype=data.dtype)
        num_points = len(data)
        data_padded[:num_points] = data
        data_padded[num_points:] = 255
        return {"points_semantics": data_padded}

    def prepare_sample_data(self):
        """Prepares sample data."""
        self.sample_data = self.join(self.sample_data, self.ego_pose, validate="1:1")
        self.sample_data = self.join(self.sample_data, self.calibrated_sensor, validate="m:1")
        self.sample_data = self.join(self.sample_data, self.sensor, validate="m:1")

    def prepare_sample_annotation(self, sample_annotation: pl.DataFrame):
        """Perpares sample annotation."""
        size = series_to_numpy(sample_annotation["sample_annotation.size"])
        size_aligned = size[..., [1, 0, 2]]  # front, left, up

        sample_annotation = sample_annotation.with_columns(
            numpy_to_series("sample_annotation.size_aligned", size_aligned)
        )

        # obj / global
        sample_annotation = self.build_sample_annotation_transforms(sample_annotation)
        return sample_annotation

    def assign_instance_index(self):
        """Assign unique index to each instance."""
        self.instance = (
            self.instance.sort("instance.token")
            .with_row_count("instance.index", offset=1)  # 0 will be used as "no instance"
            .with_columns(pl.col("instance.index").cast(pl.Int64))
        )

    def _prepare(self, key_frames_only: bool = True):
        """Prepare dataset for usage."""
        self.sample = self._assign_sample_keyframe_index(sample=self.sample)
        self.assign_instance_index()
        if key_frames_only:
            self.filter_key_frames_only()
        self.build_transformations()
        self.prepare_sample_data()
        if self.with_sample_annotation:
            self.sample_annotation = self.prepare_sample_annotation(self.sample_annotation)
        self.build_sample_data_dict()

    def sample_scene(self):
        """Samples and returns a random scene."""
        return self.get_scene(self.scene.sample(1))

    def get_scene(self, scene: pl.DataFrame):
        """Get samples for a given scene."""
        return self.join(scene, self.sample)

    def load_cameras_and_lidar(
        self,
        scene: pl.DataFrame,
        cameras: Sequence[CAMERA_CHANNELS] = CAMERAS,
        lidar: LIDAR_CHANNELS = "LIDAR_TOP",
    ):
        """Loads camera and lidar data for a given scene."""
        scene = self.load_sample_data(scene, lidar, with_data=True)
        for cam in cameras:
            scene = self.load_sample_data(scene, cam, with_data=True)
        return scene

    def scene_by_name(self, scene_name: str):
        """Select scene by name."""
        scene = self.get_scene(self.scene.filter(pl.col("scene.name") == scene_name))
        return scene

    def load_scene(self, scene_name: str):
        """Loads a scene by name."""
        scene = self.scene_by_name(scene_name)
        scene = self.load_cameras_and_lidar(scene)
        scene = self.load_lidarseg(scene, with_data=True)
        scene = scene.sort("sample.timestamp")
        return scene

    def train_scenes(self):
        """Training scenes."""
        import nuscenes.utils.splits as splits

        if self.version == "v1.0-mini":
            scene_names = pl.Series("scene.name", splits.mini_train)
        elif self.version == "v1.0-trainval":
            scene_names = pl.Series("scene.name", splits.train)
        else:
            raise ValueError("Invalid version")
        return self.scene.filter(pl.col("scene.name").is_in(scene_names))

    def val_scenes(self):
        """Validation scenes."""
        import nuscenes.utils.splits as splits

        if self.version == "v1.0-mini":
            scene_names = pl.Series("scene.name", splits.mini_val)
        elif self.version == "v1.0-trainval":
            scene_names = pl.Series("scene.name", splits.val)
        else:
            raise ValueError("Invalid version")
        return self.scene.filter(pl.col("scene.name").is_in(scene_names))

    def scenes_by_split(self, split: Literal["train", "val", "trainval"]):
        """Scenes in selected splits."""
        if split == "train":
            return self.train_scenes()
        elif split == "val":
            return self.val_scenes()
        elif split == "trainval":
            return pl.concat([self.train_scenes(), self.val_scenes()], how="vertical")
        else:
            raise ValueError("Invalid split")
