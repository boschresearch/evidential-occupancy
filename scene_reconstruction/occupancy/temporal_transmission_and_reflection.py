"""Temporal accumulation of transmission and reflection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
import torch
import tqdm
from torch import Tensor

from scene_reconstruction.core import einsum_transform
from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series
from scene_reconstruction.data.nuscenes.scene_utils import scene_to_tensor_dict
from scene_reconstruction.math.spherical_coordinate_system import (
    cartesian_to_spherical,
    spherical_volume_element_center_and_voxel_size,
)


def _load_flow_and_rt(ds: NuscenesDataset, scene: pl.DataFrame, extra_data_root: Union[Path, str]):
    scene = ds.load_reflection_and_transmission_spherical(scene, extra_data_root)
    scene = ds.load_scene_flow_polars(scene, extra_data_root)
    return scene


class AccumulateFrames:
    """Accumlate reflection and transmission for reference frame."""

    def __init__(
        self,
        ds: NuscenesDataset,
        ref_frame: pl.DataFrame,
        extra_data_root: Union[Path, str],
        icp_alignment: bool = False,
        batch_size: int = 2,
        max_num_frames: int = 50,
        max_ego_pose_difference: float = 20.0,
        device: str = "cuda",
        num_threads: int = 8,
    ) -> None:
        """Initialize refernce frame."""
        self.extra_data_root = extra_data_root
        self.kwargs = {"device": device, "non_blocking": True}
        self.ds = ds
        self.use_icp_alignment = icp_alignment
        self.frame_dict = {k: v.to(**self.kwargs) for k, v in self._load_fn(ref_frame).items()}
        self.instance_ids = self.frame_dict["frame_instance_ids"]
        self.instance_from_global = self.frame_dict["frame_instance_from_global"]
        self.global_from_ego = series_to_torch(ref_frame["LIDAR_TOP.transform.global_from_ego"]).to(**self.kwargs)
        self.instance_from_ego = torch.einsum("bnig,bge->bnie", self.instance_from_global, self.global_from_ego)
        self.sph_volume = Volume(
            lower=series_to_torch(ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.lower"]).to(
                **self.kwargs
            ),
            upper=series_to_torch(ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.upper"]).to(
                **self.kwargs
            ),
        )
        self.ego_volume = Volume(
            lower=series_to_torch(ref_frame["LIDAR_TOP.scene_flow.volume.lower"]).to(**self.kwargs),
            upper=series_to_torch(ref_frame["LIDAR_TOP.scene_flow.volume.upper"]).to(**self.kwargs),
        )

        self.max_ego_pose_difference = max_ego_pose_difference
        self.max_num_frames = max_num_frames
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._init_ref_frame()

    def _init_ref_frame(self):
        sample_points = self._generate_sample_points(
            self.frame_dict["frame_instance_ids"],
            self.frame_dict["frame_instance_from_global"],
            self.frame_dict["frame_ego_from_global"],
            self.frame_dict["frame_lidar_from_ego"],
        )
        sample_points_spherical = cartesian_to_spherical(sample_points)
        spherical_rt = self.frame_dict["spherical_rt"]
        cartesian_rt = self.sph_volume.sample_volume(spherical_rt, sample_points_spherical)
        spherical_voxel_size = self.sph_volume.voxel_size(spherical_rt)
        cart_voxel_size = self.ego_volume.voxel_size(cartesian_rt)
        # there might be points containing "nan" values, since they have no valid transformation
        scale = (
            cart_voxel_size.prod(-1)[:, None, None, None]
            / spherical_volume_element_center_and_voxel_size(
                sample_points_spherical, spherical_voxel_size[:, None, None, None, :]
            )
        ).nan_to_num(0.0)
        cartesian_rt_scaled = scale.unsqueeze(1) * cartesian_rt
        assert cartesian_rt_scaled.shape[0] == 1, "Only one ref frame allowed"
        self.agg_rt = cartesian_rt_scaled.sum(0, keepdim=True)
        self.num_samples = cartesian_rt_scaled.shape[0]

    # def _icp_alignment(
    #     self,
    #     ref_points_global: Tensor,
    #     ref_points_global_mask: Tensor,
    #     new_points_global: Tensor,
    #     new_points_global_mask: Tensor,
    # ):
    #     # ref_points_global: [1, N, 3]
    #     # new_points_global: [B, N, 3]
    #     # new_points_global_mask: [B, N]
    #     ref_points = ref_points_global[ref_points_global_mask]  # [M, 3]
    #     ref_from_global_list = []
    #     for points_, mask_ in zip(new_points_global, new_points_global_mask):
    #         points_global = points_[mask_]  # [N, 3]
    #         ref_from_global = register_frame(ref_points, points_global)
    #         ref_from_global_list.append(ref_from_global)
    #     ref_from_global = torch.stack(ref_from_global_list)
    #     return ref_from_global

    # def _icp_ref_from_global(self, scene: pl.DataFrame):
    #     points_lidar = series_to_torch(scene["LIDAR_TOP.sample_data.points_lidar"]).to(**self.kwargs)
    #     points_mask = series_to_torch(scene["LIDAR_TOP.sample_data.points_mask"]).to(**self.kwargs)
    #     global_from_ego = series_to_torch(scene["LIDAR_TOP.transform.global_from_ego"]).to(**self.kwargs)
    #     ego_from_lidar = series_to_torch(scene["LIDAR_TOP.transform.ego_from_sensor"]).to(**self.kwargs)
    #     global_from_lidar = global_from_ego @ ego_from_lidar
    #     points_global = einsum_transform("bgl,bnl->bng", global_from_lidar, points=points_lidar)
    #     ref_from_global = self._icp_alignment(self.points_global, self.points_mask, points_global, points_mask)
    #     return ref_from_global

    def _generate_sample_points(
        self,
        frame_instance_ids: Tensor,
        frame_instance_from_global: Tensor,
        frame_ego_from_global: Tensor,
        frame_lidar_from_ego: Tensor,
    ):
        frame_global_from_instance = frame_instance_from_global.inverse()
        if self.use_icp_alignment:
            raise NotImplementedError()

        # build transformations
        frame_lidar_from_global = frame_lidar_from_ego @ frame_ego_from_global
        frame_ego_from_instance = torch.einsum("beg,bngi->bnei", frame_ego_from_global, frame_global_from_instance)
        frame_lidar_from_instance = torch.einsum("blg,bngi->bnli", frame_lidar_from_global, frame_global_from_instance)

        frame_ego_from_self_ego = frame_ego_from_instance @ self.instance_from_ego
        frame_lidar_from_self_ego = frame_lidar_from_instance @ self.instance_from_ego
        self_instance_ids_flat = self.instance_ids.flatten(1)

        frame_ego_from_self_ego_gathered_flat = frame_ego_from_self_ego.gather(
            1, self_instance_ids_flat[..., None, None].expand(frame_instance_ids.shape[0], -1, 4, 4)
        )
        frame_ego_from_self_ego_dense = frame_ego_from_self_ego_gathered_flat.view(
            -1, *self.instance_ids.shape[1:], 4, 4
        )
        frame_ego_points = einsum_transform(
            "bxyzfs,bxyzs->bxyzf", frame_ego_from_self_ego_dense, points=self.ego_volume.coord_grid(frame_instance_ids)
        )
        # sample ids to check for valid transforms
        self_instance_ids_sampled = self.ego_volume.sample_volume(
            frame_instance_ids.unsqueeze(1).float(), frame_ego_points, mode="nearest", fill_invalid=float("nan")
        )
        valid_transform = self_instance_ids_sampled.squeeze(1) == self.instance_ids.float()  # [B, X, Y, Z]

        frame_lidar_from_self_ego_gathered_flat = frame_lidar_from_self_ego.gather(
            1, self_instance_ids_flat[..., None, None].expand(frame_instance_ids.shape[0], -1, 4, 4)
        )
        frame_lidar_from_self_ego_dense = frame_lidar_from_self_ego_gathered_flat.view(
            -1, *self.instance_ids.shape[1:], 4, 4
        )
        frame_lidar_points = einsum_transform(
            "bxyzfs,bxyzs->bxyzf",
            frame_lidar_from_self_ego_dense,
            points=self.ego_volume.coord_grid(frame_instance_ids),
        )

        frame_lidar_points_valid = torch.where(valid_transform[..., None], frame_lidar_points, float("nan"))

        return frame_lidar_points_valid

    def _load_fn(self, frame: pl.DataFrame):
        frame = _load_flow_and_rt(self.ds, frame, self.extra_data_root)
        frame_dict = scene_to_tensor_dict(
            frame,
            {
                "frame_instance_ids": "LIDAR_TOP.scene_flow.scene_instance_index",
                "frame_instance_from_global": "LIDAR_TOP.scene_flow.instance_from_global",
                "frame_ego_from_global": "LIDAR_TOP.transform.ego_from_global",
                "frame_lidar_from_ego": "LIDAR_TOP.transform.sensor_from_ego",
                "spherical_rt": "LIDAR_TOP.reflection_and_transmission_spherical",
            },
        )
        return frame_dict

    def _accumulate_frames(
        self,
        frame_instance_ids: Tensor,
        frame_instance_from_global: Tensor,
        frame_ego_from_global: Tensor,
        frame_lidar_from_ego: Tensor,
        spherical_rt: Tensor,
    ):
        sample_points = self._generate_sample_points(
            frame_instance_ids,
            frame_instance_from_global,
            frame_ego_from_global,
            frame_lidar_from_ego,
        )
        sample_points_spherical = cartesian_to_spherical(sample_points)
        cartesian_rt = self.sph_volume.sample_volume(spherical_rt, sample_points_spherical)
        spherical_voxel_size = self.sph_volume.voxel_size(spherical_rt)
        cart_voxel_size = self.ego_volume.voxel_size(cartesian_rt)
        # there might be points containing "nan" values, since they have no valid transformation
        scale = (
            cart_voxel_size.prod(-1)[:, None, None, None]
            / spherical_volume_element_center_and_voxel_size(
                sample_points_spherical, spherical_voxel_size[:, None, None, None, :]
            )
        ).nan_to_num(0.0)
        cartesian_rt_scaled = scale.unsqueeze(1) * cartesian_rt

        self.agg_rt += cartesian_rt_scaled.sum(0, keepdim=True)
        self.num_samples += cartesian_rt_scaled.shape[0]

    def process_frames_in_radius(self, scene: pl.DataFrame):
        """Process frames in radius."""
        scene = scene.sort("LIDAR_TOP.sample_data.timestamp")
        frame_ego_from_global = series_to_torch(scene["LIDAR_TOP.transform.ego_from_global"])
        frame_ego_from_self_ego = frame_ego_from_global @ self.global_from_ego.cpu()
        pos_diff = frame_ego_from_self_ego[:, :3, 3].norm(dim=-1)
        frams_in_radius = pos_diff < self.max_ego_pose_difference
        scene = scene.with_columns(torch_to_series("frame_in_radius", frams_in_radius))
        scene = scene.filter(pl.col("frame_in_radius"))
        if len(scene) > self.max_num_frames:
            indices = torch.linspace(-0.5, len(scene) + 0.5, self.max_num_frames).round().long()
            scene = scene.with_row_count("index").filter(
                pl.col("index").is_in(torch_to_series("select_indices", indices))
            )

        items = scene.select(
            "LIDAR_TOP.transform.ego_from_global",
            "LIDAR_TOP.transform.sensor_from_ego",
            "scene.name",
            "LIDAR_TOP.sample_data.token",
        ).iter_slices(self.batch_size)
        for sample in items:
            frame_dict = self._load_fn(sample)
            frame_dict = {k: v.to(**self.kwargs) for k, v in frame_dict.items()}
            self._accumulate_frames(**frame_dict)


@dataclass
class TemporalTransmissionAndReflection:
    """Temporal accumulation of reflection and transmission using scene flow info."""

    ds: NuscenesDataset
    extra_data_root: Path
    reference_keyframes_only: bool = True
    frame_accumulation_kwargs: dict[str, Any] = field(default_factory=dict)
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def save_accumulated_sample(self, scene_name: str, sample: pl.DataFrame):
        """Saves a single sample."""
        assert len(sample) == 1
        filename = self.save_path(scene_name, sample)
        filename.parent.mkdir(exist_ok=True, parents=True)
        # remove batch dim
        sample.write_ipc(filename, compression="zstd")

    def save_path(self, scene_name: str, sample: pl.DataFrame):
        """Save path."""
        assert len(sample) == 1
        filename = (
            Path(self.extra_data_root)
            / "reflection_and_transmission_multi_frame"
            / scene_name
            / "LIDAR_TOP"
            / f"{sample.item(0, 'LIDAR_TOP.sample_data.token')}.arrow"
        )
        return filename

    def process_scene(self, scene: pl.DataFrame):
        """Processes single scene."""
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene = self.ds.sort_by_time(scene)

        if self.reference_keyframes_only:
            reference_frames = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
        else:
            reference_frames = scene
        for reference_frame in tqdm.tqdm(reference_frames.iter_slices(1), total=len(reference_frames), position=1):
            if self.missing_only:
                filename = self.save_path(reference_frame["scene.name"].item(), reference_frame)
                if filename.exists():
                    continue

            reference_frame = self.ds.load_reflection_and_transmission_spherical(reference_frame, self.extra_data_root)
            reference_frame = self.ds.load_scene_flow_polars(reference_frame, self.extra_data_root)

            agg_frames = AccumulateFrames(
                self.ds,
                ref_frame=reference_frame,
                extra_data_root=self.extra_data_root,
                **self.frame_accumulation_kwargs,
            )

            agg_frames.process_frames_in_radius(scene)
            assert agg_frames.agg_rt is not None, "No frames accumulated"
            agg_frames_mean = agg_frames.agg_rt / agg_frames.num_samples
            data = (
                reference_frame.select(
                    "LIDAR_TOP.sample_data.token",
                    "LIDAR_TOP.scene_flow.volume.lower",
                    "LIDAR_TOP.scene_flow.volume.upper",
                )
                .rename(
                    {
                        "LIDAR_TOP.scene_flow.volume.lower": "LIDAR_TOP.reflection_and_transmission_multi_frame.volume.lower",
                        "LIDAR_TOP.scene_flow.volume.upper": "LIDAR_TOP.reflection_and_transmission_multi_frame.volume.upper",
                    }
                )
                .with_columns(
                    torch_to_series("LIDAR_TOP.reflection_and_transmission_multi_frame", agg_frames_mean.cpu())
                )
            )
            self.save_accumulated_sample(reference_frame["scene.name"].item(), data)

    def process_data(self) -> None:
        """Process dataset."""
        # load all sample data with lidar token
        scene_to_process = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in (tbar := tqdm.tqdm(scene_to_process.iter_slices(1), total=len(scene_to_process), position=0)):
            scene_name = scene["scene.name"].item()
            tbar.set_description_str(f"Processing {scene_name}")
            self.process_scene(scene)
