"""Scene flow."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tqdm
from torch import Tensor

from scene_reconstruction.core.transform import einsum_transform
from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import nested_to_torch, series_to_torch, torch_to_series


def build_scene_instance_motion(
    ds: NuscenesDataset,
    scene: pl.DataFrame,
    scene_with_annos: pl.DataFrame,
    scene_instance_index: pl.DataFrame,
):
    """Aggregate motion information per instance."""
    scene_matrix = scene_with_annos.with_columns(
        pl.col("transform.obj_from_global").cast(pl.List(pl.List(pl.Float32)))
    ).pivot(
        index="LIDAR_TOP.sample_data.token",
        columns="scene_instance.index",
        values="transform.obj_from_global",
    )
    scene_matrix_token = scene_matrix.select("LIDAR_TOP.sample_data.token")
    scene_matrix_data = scene_matrix.select([str(i) for i in scene_instance_index["scene_instance.index"].to_list()])
    scene_obj_from_global = nested_to_torch(
        scene_matrix_data.to_numpy(), fill_value=np.full((4, 4), float("nan"), dtype=np.float32)
    )  # [time index, scene instance index, 4, 4]
    scene_obj_from_global_valid_mask = nested_to_torch(
        scene_matrix_data.select(pl.all().is_not_null()).to_numpy()
    )  # [time index, scene instance index]

    # static transform for reserved instance id 0

    scene_obj_from_global = torch.cat(
        [torch.eye(4)[None, None].expand_as(scene_obj_from_global[:, :1]), scene_obj_from_global], 1
    )
    scene_obj_from_global_valid_mask = torch.cat(
        [torch.ones_like(scene_obj_from_global_valid_mask[:, :1]), scene_obj_from_global_valid_mask], 1
    )

    scene_matrix = scene_matrix_token.with_columns(
        torch_to_series("LIDAR_TOP.scene_flow.instance_from_global", scene_obj_from_global),
        torch_to_series("LIDAR_TOP.scene_flow.instance_from_global_mask", scene_obj_from_global_valid_mask),
    )
    scene_matrix_token_without_annos = scene.select("LIDAR_TOP.sample_data.token").filter(
        pl.col("LIDAR_TOP.sample_data.token").is_in(scene_matrix["LIDAR_TOP.sample_data.token"]).not_()
    )
    num_samples_without_annos = len(scene_matrix_token_without_annos)

    if num_samples_without_annos > 0:
        num_instances = scene_obj_from_global.shape[1]
        scene_obj_from_global_without_annos = torch.cat(
            [
                torch.eye(4)[None, None].expand(num_samples_without_annos, -1, -1, -1),
                torch.full((num_samples_without_annos, num_instances - 1, 4, 4), fill_value=float("nan")),
            ],
            1,
        )
        scene_obj_from_global_valid_mask_without_annos = torch.cat(
            [
                torch.ones(num_samples_without_annos, 1, dtype=torch.bool),
                torch.zeros(num_samples_without_annos, num_instances - 1, dtype=torch.bool),
            ],
            1,
        )
        scene_matrix_without_annos = scene_matrix_token_without_annos.with_columns(
            torch_to_series("LIDAR_TOP.scene_flow.instance_from_global", scene_obj_from_global_without_annos),
            torch_to_series(
                "LIDAR_TOP.scene_flow.instance_from_global_mask", scene_obj_from_global_valid_mask_without_annos
            ),
        )
        if len(scene_matrix) > 0:
            scene_matrix = pl.concat([scene_matrix, scene_matrix_without_annos], how="vertical")
        else:
            scene_matrix = scene_matrix_without_annos

    scene_instance_motion = ds.join(scene, scene_matrix)
    assert len(scene) == len(scene_instance_motion)

    return scene_instance_motion


def extract_instance_ids_chunked(
    lidar_token: pl.DataFrame,
    sample: pl.DataFrame,
    ego_volume: Volume,
    volume_shape: Sequence[int],
    max_points: int = 200000,
    **kwargs,
):
    """Extract instance ids for each grid point."""
    obj_from_global = series_to_torch(sample["transform.obj_from_global"]).to(**kwargs)
    obj_bbox = series_to_torch(sample["sample_annotation.size_aligned"]).to(**kwargs)
    global_from_ego = series_to_torch(sample["LIDAR_TOP.transform.global_from_ego"]).to(**kwargs)
    instance_index = series_to_torch(sample["scene_instance.index"]).to(**kwargs)
    obj_from_ego = obj_from_global @ global_from_ego
    num_points = math.prod(volume_shape)
    num_chunks = (num_points + max_points - 1) // max_points
    grid_points = ego_volume.to(**kwargs).new_coord_grid(volume_shape)
    # chunk along x dim
    instance_ids_chunked = []
    for grid_points_chunk in grid_points.chunk(num_chunks, dim=1):
        grid_coordinates_obj = einsum_transform("boe,bxyze->bxyzo", obj_from_ego, points=grid_points_chunk.to(**kwargs))
        bbox_half = 0.5 * obj_bbox[:, None, None, None, :]
        distance_to_center = grid_coordinates_obj.norm(dim=-1)
        inside_box = (grid_coordinates_obj.abs() <= bbox_half).all(-1)
        soft_box_distance = torch.where(inside_box, distance_to_center, float("inf"))
        assigned_bbox = soft_box_distance.argmin(0)
        in_any_bbox = inside_box.sum(0) > 0
        instance_ids = torch.where(in_any_bbox, instance_index[assigned_bbox], 0)  # 0 = static background
        instance_ids_chunked.append(instance_ids.cpu())
    instance_ids_combined = torch.cat(instance_ids_chunked)
    assert instance_ids_combined.shape == volume_shape
    lidar_token = lidar_token.with_columns(
        torch_to_series("LIDAR_TOP.scene_flow.scene_instance_index", instance_ids_combined[None])
    )
    return lidar_token


def extract_instance_ids(
    lidar_token: pl.DataFrame, sample: pl.DataFrame, ego_volume: Volume, volume_shape: Sequence[int]
):
    """Extract instance ids for each grid point."""
    obj_from_global = series_to_torch(sample["transform.obj_from_global"]).cuda()
    obj_bbox = series_to_torch(sample["sample_annotation.size_aligned"]).cuda()
    global_from_ego = series_to_torch(sample["LIDAR_TOP.transform.global_from_ego"]).cuda()
    instance_index = series_to_torch(sample["scene_instance.index"]).cuda()
    obj_from_ego = obj_from_global @ global_from_ego
    grid_coordinates_obj = einsum_transform(
        "boe,bxyze->bxyzo", obj_from_ego, points=ego_volume.new_coord_grid(volume_shape).cuda()
    )
    bbox_half = 0.5 * obj_bbox[:, None, None, None, :]
    distance_to_center = grid_coordinates_obj.norm(dim=-1)
    inside_box = (grid_coordinates_obj.abs() <= bbox_half).all(-1)
    soft_box_distance = torch.where(inside_box, distance_to_center, float("inf"))
    assigned_bbox = soft_box_distance.argmin(0)
    in_any_bbox = inside_box.sum(0) > 0
    instance_ids = torch.where(in_any_bbox, instance_index[assigned_bbox], 0)  # 0 = static background
    lidar_token = lidar_token.with_columns(torch_to_series("scene_flow.scene_instance_index", instance_ids[None]))
    return lidar_token


def lookup_instance_transform(
    instance_ids: Tensor,
    instance_from_tgt: Tensor,
    instance_from_src: Tensor,
    instance_from_tgt_mask: Tensor,
    instance_from_src_mask: Tensor,
):
    """Lookup transforms for each grid cell."""
    # instance_ids: B, X, Y, Z
    # instance_from_tgt, instance_from_src: B, X, Y, Z, 4, 4
    # instance_from_tgt_mask, instance_from_src_mask: B, X, Y, Z
    # instance needs to be part of both frames transform
    # fall back to static transform
    instance_ids_flat = instance_ids.flatten(1)
    tgt_mask = instance_from_tgt_mask.gather(1, instance_ids_flat)
    src_mask = instance_from_src_mask.gather(1, instance_ids_flat)
    valid_instance_ids_flat = torch.where(
        tgt_mask & src_mask,
        instance_ids_flat,
        0,
    )
    tgt_from_src = instance_from_tgt.inverse() @ instance_from_src
    tgt_from_src_gathered_flat = tgt_from_src.gather(1, valid_instance_ids_flat[..., None, None].expand(-1, -1, 4, 4))
    tgt_from_src_transform = tgt_from_src_gathered_flat.view(*instance_ids.shape, 4, 4)

    return tgt_from_src_transform


@dataclass
class SceneFlow:
    """Scene flow from boxes."""

    ds: NuscenesDataset
    extra_data_root: Path

    cartesian_lower: tuple[float, float, float] = (-40.0, -40.0, -1.0)
    cartesian_upper: tuple[float, float, float] = (40.0, 40.0, 5.4)
    cartesian_shape: tuple[int, int, int] = (400, 400, 32)

    device: str = "cuda"
    missing_only: bool = False

    def load_scene(self, scene_name: str):
        """Loads a single scene for processing."""
        # load single scene
        scene = self.ds.scene_by_name(scene_name)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene_with_annos = self.ds.join(scene, self.ds.sample_annotation_dict["LIDAR_TOP"])
        scene_with_annos = self.ds.join(scene_with_annos, self.ds.instance)
        # unique instance id per scene starting from 1 (0 reserved for static world)
        scene_instance_index = (
            scene_with_annos.select("instance.token")
            .unique()
            .sort("instance.token")
            .with_row_count("scene_instance.index", offset=1)
            .with_columns(pl.col("scene_instance.index").cast(pl.Int64))
        )
        scene_with_annos = self.ds.join(scene_with_annos, scene_instance_index)
        return scene, scene_with_annos, scene_instance_index

    def save_scene_flow_polars(self, scene_name: str, sample: pl.DataFrame):
        """Saves a single sample."""
        assert len(sample) == 1
        filename = self.save_path(scene_name, sample)
        filename.parent.mkdir(exist_ok=True, parents=True)
        # remove batch dim
        sample.write_ipc(filename, compression="zstd")

    def save_path(self, scene_name, sample: pl.DataFrame):
        """Save from scene name and lidar token."""
        assert len(sample) == 1
        filename = (
            Path(self.extra_data_root)
            / "scene_flow"
            / scene_name
            / "LIDAR_TOP"
            / f"{sample.item(0, 'LIDAR_TOP.sample_data.token')}.arrow"
        )
        return filename

    def process_scene(self, scene_name: str):
        """Process a single scene."""
        scene, scene_with_annos, scene_instance_index = self.load_scene(scene_name)
        scene = build_scene_instance_motion(self.ds, scene, scene_with_annos, scene_instance_index)
        ego_volume = Volume.new_volume(lower=self.cartesian_lower, upper=self.cartesian_upper)
        volume_lower = torch_to_series("LIDAR_TOP.scene_flow.volume.lower", ego_volume.lower)
        volume_upper = torch_to_series("LIDAR_TOP.scene_flow.volume.upper", ego_volume.upper)

        for sample in tqdm.tqdm(scene.iter_slices(1), total=len(scene), position=1):
            if self.missing_only:
                filename = self.save_path(scene_name, sample)
                if filename.exists():
                    continue
            # sample has annotations
            if sample["LIDAR_TOP.sample_data.token"].is_in(scene_with_annos["LIDAR_TOP.sample_data.token"]).item():
                sample_with_annos = self.ds.join(sample.select("LIDAR_TOP.sample_data.token"), scene_with_annos)
                sample_with_ids = extract_instance_ids_chunked(
                    sample, sample_with_annos, ego_volume, self.cartesian_shape, device=self.device
                )
            else:
                # everything is background
                sample_with_ids = sample.select("LIDAR_TOP.sample_data.token").with_columns(
                    torch_to_series(
                        "LIDAR_TOP.scene_flow.scene_instance_index",
                        torch.zeros(1, *self.cartesian_shape, dtype=torch.long),
                    )
                )

            sample = self.ds.join(
                sample.select(
                    "LIDAR_TOP.sample_data.token",
                    "LIDAR_TOP.scene_flow.instance_from_global",
                    "LIDAR_TOP.scene_flow.instance_from_global_mask",
                ),
                sample_with_ids.select(
                    "LIDAR_TOP.sample_data.token",
                    "LIDAR_TOP.scene_flow.scene_instance_index",
                ),
            )
            sample = sample.with_columns(volume_lower, volume_upper)
            self.save_scene_flow_polars(scene_name, sample)

        return scene

    def process_data(self):
        """Process dataset."""
        self.ds.load_sensor_sample_annotation("LIDAR_TOP")
        for scene_name in tqdm.tqdm(self.ds.scene["scene.name"], position=0, desc="Calculate scene flow from boxes"):
            self.process_scene(scene_name)


def deform_grid(scene: pl.DataFrame, volume: Volume):
    """Deform grid considering dynmic objects."""
    instance_ids = series_to_torch(scene["LIDAR_TOP.scene_flow.scene_instance_index"])
    instance_from_global = series_to_torch(scene["LIDAR_TOP.scene_flow.instance_from_global"])
    global_from_ego = series_to_torch(scene["LIDAR_TOP.transform.global_from_ego"])
    instance_from_ego = torch.einsum("bnig,bge->bnie", instance_from_global, global_from_ego)
    instance_from_ego_src = instance_from_ego[1:]
    instance_from_ego_tgt = instance_from_ego[:-1]

    instance_from_mask = series_to_torch(scene["LIDAR_TOP.scene_flow.instance_from_global_mask"])
    instance_from_src_mask = instance_from_mask[1:]
    instance_from_tgt_mask = instance_from_mask[:-1]
    instance_ids_src = instance_ids[1:]
    tgt_from_src = lookup_instance_transform(
        instance_ids_src,
        instance_from_ego_tgt,
        instance_from_ego_src,
        instance_from_tgt_mask,
        instance_from_src_mask,
    )
    new_grid_points = volume.new_coord_grid(instance_ids.shape[-3:])
    warped_grid_points = einsum_transform("bxyzon,bxyzn->bxyzo", tgt_from_src, points=new_grid_points)
    return warped_grid_points
