"""Lidar rendering."""
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import einops
import kaolin
import polars as pl
import torch
import torch.nn as nn
import tqdm
from kaolin.ops.spc.points import morton_to_points, points_to_morton, unbatched_points_to_octree
from kaolin.rep.spc import Spc
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.collections import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from typing_extensions import Literal

from scene_reconstruction.core import einsum_transform
from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.scene_utils import scene_to_tensor_dict
from scene_reconstruction.math.dempster_shafer import belief_from_reflection_and_transmission_stacked


def unbatched_spc_from_volume(features: Tensor, volume: Volume, mask_value=0):
    """Converts volume to structured point cloud."""
    assert features.shape[0] == 1
    assert features.is_cuda
    volume_grid = volume.coord_grid(features)  # [1, X, Y, Z, 3]
    # max next power of 2 for octree level
    level = max([int(math.ceil(math.log2(x))) for x in volume.volume_shape(features)])
    next_power_of_2 = 2**level
    volume_shape = torch.tensor([volume.volume_shape(features)], device=features.device)
    normed_limits = volume_shape / next_power_of_2

    # spc voxel index coords, range from [0, 0, 0] to [X, Y, Z]
    lower_voxel_spc = torch.zeros_like(volume.lower)
    upper_voxel_spc = volume_shape
    spc_voxel_volume = Volume(lower=lower_voxel_spc, upper=upper_voxel_spc)
    spc_voxel_volume_from_ego = spc_voxel_volume.self_from_other(volume)
    grid_points_spc_voxel_volume = einsum_transform("bse,bxyze->bxyzs", spc_voxel_volume_from_ego, points=volume_grid)
    grid_points_quantized = torch.floor(torch.clamp(grid_points_spc_voxel_volume, 0, next_power_of_2 - 1)).short()

    # spc normalized coords, range from -1.0 to 1.0
    lower_norm_spc = -torch.ones_like(volume.lower)
    upper_norm_spc = -torch.ones_like(volume.upper) + 2 * normed_limits
    spc_norm_volume = Volume(lower=lower_norm_spc, upper=upper_norm_spc)
    spc_norm_volume_from_ego = spc_norm_volume.self_from_other(volume)
    assert (
        spc_norm_volume_from_ego[0, 0, 0] == spc_norm_volume_from_ego[0, 1, 1] == spc_norm_volume_from_ego[0, 2, 2]
    ), "voxel size must be uniform across dims"
    scale = 1 / spc_norm_volume_from_ego[0, 0, 0].item()

    non_zero_mask = (features != mask_value).any(1)  # [B, X, Y, Z]

    features_flat = einops.rearrange(features, "b c x y z -> b x y z c")[non_zero_mask]  # [N, C]
    points_quantized_flat = grid_points_quantized[non_zero_mask]  # [N, 3]

    # sort points and features
    morton, keys = torch.sort(points_to_morton(points_quantized_flat.contiguous()).contiguous())
    points_quantized_flat = morton_to_points(morton.contiguous())
    octree = unbatched_points_to_octree(points_quantized_flat, level, sorted=True)
    features_flat = features_flat[keys]

    # A full SPC requires octree hierarchy + auxilary data structures
    lengths = torch.tensor([len(octree)], dtype=torch.int32)  # Single entry batch
    return (
        Spc(octrees=octree, lengths=lengths, features=features_flat, max_level=level),
        spc_norm_volume_from_ego[0],
        scale,
    )


def unbatched_raytrace_lidar_transmissions(
    occupancies: Tensor,  # [B, C, X, Y, Z]
    volume: Volume,
    start_points: Tensor,  # [N, 3]
    end_points: Tensor,  # [N, 3]
    mask_value=0,
):
    """Render lidar in occupancy volume."""
    spc, spc_from_ego, scale = unbatched_spc_from_volume(occupancies, volume=volume, mask_value=mask_value)
    start_points_spc = einsum_transform("se,ne->ns", spc_from_ego, points=start_points)
    end_points_spc = einsum_transform("se,ne->ns", spc_from_ego, points=end_points)
    ray_direction = end_points_spc - start_points_spc
    ray_length: Tensor = ray_direction.norm(dim=-1, keepdim=True)
    ray_direction_normalized = ray_direction / ray_length
    num_rays = start_points_spc.shape[0]
    ridx: Tensor
    pidx: Tensor
    depth: Tensor
    ridx, pidx, depth = kaolin.render.spc.unbatched_raytrace(  # type: ignore
        spc.octrees,
        spc.point_hierarchies,
        spc.pyramids[0],
        spc.exsum,
        start_points_spc + 2**-spc.max_level,  # this offset seems to fix points at [0.0, 0.0, 0.0] bug?
        ray_direction_normalized,
        spc.max_level,
        return_depth=True,
        with_exit=False,
    )

    first_hits_mask: Tensor = kaolin.render.spc.mark_pack_boundaries(ridx)
    first_hit_ridx = ridx[first_hits_mask]
    first_hit_depth = torch.full((num_rays, 1), fill_value=float("inf"), device=depth.device)
    first_hit_depth.index_copy_(0, first_hit_ridx.long(), depth[first_hits_mask])

    before_lidar_reflection = (depth < ray_length.index_select(0, ridx)).float()
    num_transmissions_per_ray = torch.full((num_rays, 1), fill_value=0.0, device=depth.device)
    num_transmissions_per_ray.index_add_(0, ridx, before_lidar_reflection)

    num_transmissions_per_voxel = torch.full_like(spc.features[:, :1], fill_value=0.0, dtype=torch.float)
    num_transmissions_per_voxel.index_add_(
        0,
        pidx - spc.pyramids[0, 1, spc.max_level],
        before_lidar_reflection,
    )

    spc_num_transmissions = Spc(
        octrees=spc.octrees,
        max_level=spc.max_level,
        lengths=spc.lengths,
        features=num_transmissions_per_voxel,
    )

    first_hit_depth_unscaled = first_hit_depth * scale

    return first_hit_depth_unscaled, num_transmissions_per_ray, spc_num_transmissions


class RenderLidarDepth(nn.Module):
    """Lidar rendering."""

    def __init__(
        self,
        volume: Volume,
        eval_volume_ego: Volume,
        free_index: int = 17,
        min_distance: float = 2.5,
        volume_frame: Literal["ego", "lidar"] = "ego",
    ) -> None:
        """Lidar rendering in ego volume."""
        super().__init__()
        self.volume = volume
        self.free_index = free_index
        self.min_distance = min_distance
        self.max_distance = (eval_volume_ego.upper - eval_volume_ego.lower).norm(dim=-1)
        self.volume_frame = volume_frame
        self.eval_volume_ego = eval_volume_ego

    def forward(
        self,
        occupancy: Tensor,
        points_lidar: Tensor,
        points_mask: Tensor,
        ego_from_lidar: Tensor,
    ):
        """Rendering."""
        if self.volume_frame == "ego":
            return self._render_in_ego_volume(occupancy, points_lidar, points_mask, ego_from_lidar)
        elif self.volume_frame == "lidar":
            return self._render_in_lidar_volume(occupancy, points_lidar, points_mask, ego_from_lidar)
        else:
            raise NotImplementedError()

    def _render_in_ego_volume(
        self, occupancy: Tensor, points_lidar: Tensor, points_mask: Tensor, ego_from_lidar: Tensor
    ):
        """Rendering in ego volume."""
        occupancy_with_channel = occupancy.unsqueeze(1)
        start_points_ego = einsum_transform(
            "bel,bnl->bne", ego_from_lidar, points=torch.zeros_like(points_lidar)
        )  # lidar origin
        end_points_ego = einsum_transform("bel,bnl->bne", ego_from_lidar, points=points_lidar)

        rendered_depths = torch.full_like(start_points_ego[..., 0], fill_value=float("nan"))
        num_transmissions = torch.full_like(start_points_ego[..., 0], fill_value=float("nan"))
        lidar_depth = points_lidar.norm(dim=-1)
        for ub_num_t, ub_rendered_depth, ub_occ, ub_start_points_ego, ub_end_points_ego, ub_masks in zip(
            num_transmissions.split(1),
            rendered_depths.split(1),
            occupancy_with_channel.split(1),
            start_points_ego.split(1),
            end_points_ego.split(1),
            points_mask.split(1),
        ):
            (
                rendered_depth,
                num_transmissions_per_ray,
                spc_num_transmissions,
            ) = unbatched_raytrace_lidar_transmissions(
                occupancies=ub_occ,
                volume=self.volume,
                start_points=ub_start_points_ego[ub_masks],
                end_points=ub_end_points_ego[ub_masks],
                mask_value=self.free_index,
            )
            ub_rendered_depth[ub_masks] = rendered_depth.squeeze(1)
            ub_num_t[ub_masks] = num_transmissions_per_ray.squeeze(1)
        # calculate rendered endpoints in ego frame to clamp them to the eval bounds
        direction = end_points_ego - start_points_ego
        direction /= direction.norm(dim=-1, keepdim=True)
        finite_depth = rendered_depths.isfinite()
        rendered_depths = rendered_depths.clamp_max(self.max_distance.unsqueeze(-1))
        rendered_end_points_ego = start_points_ego + direction * rendered_depths.unsqueeze(-1)
        rendered_end_points_ego_clamped = self.eval_volume_ego.clamp_points_along_line(
            rendered_end_points_ego, start_points_ego
        )
        # recalculate depth
        rendered_depths = (rendered_end_points_ego_clamped - start_points_ego).norm(dim=-1)
        points_in_volume = (
            # ego volume check
            (end_points_ego >= self.eval_volume_ego.lower.unsqueeze(1)).all(-1)
            & (end_points_ego <= self.eval_volume_ego.upper.unsqueeze(1)).all(-1)
            & (start_points_ego >= self.eval_volume_ego.lower.unsqueeze(1)).all(-1)
            & (start_points_ego <= self.eval_volume_ego.upper.unsqueeze(1)).all(-1)
            # lidar points check
            & points_mask
            & (lidar_depth > self.min_distance)
        )  # [B, X, Y, Z]

        finite_depth = finite_depth & points_mask & (lidar_depth > self.min_distance)

        out_dict = {
            "rendered_depth": rendered_depths,
            "num_transmission": num_transmissions,
            "lidar_depth": lidar_depth,
            "points_in_ego_volume": points_in_volume,
            "finite_depth": finite_depth,
        }

        return out_dict

    def _render_in_lidar_volume(
        self, occupancy: Tensor, points_lidar: Tensor, points_mask: Tensor, ego_from_lidar: Tensor
    ):
        """Rendering in lidar volume."""
        occupancy_with_channel = occupancy.unsqueeze(1)
        start_points_lidar = torch.zeros_like(points_lidar)  # lidar origin
        end_points_lidar = points_lidar

        rendered_depths = torch.full_like(start_points_lidar[..., 0], fill_value=float("nan"))
        num_transmissions = torch.full_like(start_points_lidar[..., 0], fill_value=float("nan"))
        lidar_depth = end_points_lidar.norm(dim=-1)
        for ub_num_t, ub_rendered_depth, ub_occ, ub_start_points_ego, ub_end_points_ego, ub_masks in zip(
            num_transmissions.split(1),
            rendered_depths.split(1),
            occupancy_with_channel.split(1),
            start_points_lidar.split(1),
            end_points_lidar.split(1),
            points_mask.split(1),
        ):
            (
                rendered_depth,
                num_transmissions_per_ray,
                spc_num_transmissions,
            ) = unbatched_raytrace_lidar_transmissions(
                occupancies=ub_occ,
                volume=self.volume,
                start_points=ub_start_points_ego[ub_masks],
                end_points=ub_end_points_ego[ub_masks],
                mask_value=self.free_index,
            )
            ub_rendered_depth[ub_masks] = rendered_depth.squeeze(1)
            ub_num_t[ub_masks] = num_transmissions_per_ray.squeeze(1)
        start_points_ego = einsum_transform(
            "bel,bnl->bne", ego_from_lidar, points=torch.zeros_like(points_lidar)
        )  # lidar origin
        end_points_ego = einsum_transform("bel,bnl->bne", ego_from_lidar, points=points_lidar)
        # calculate rendered endpoints in ego frame to clamp them to the eval bounds
        finite_depth = rendered_depths.isfinite()
        rendered_depths = rendered_depths.clamp_max(self.max_distance.unsqueeze(-1))
        direction = end_points_ego - start_points_ego
        direction /= direction.norm(dim=-1, keepdim=True)
        rendered_end_points_ego = start_points_ego + direction * rendered_depths.unsqueeze(-1)
        rendered_end_points_ego_clamped = self.eval_volume_ego.clamp_points_along_line(
            rendered_end_points_ego, start_points_ego
        )
        # recalculate depth
        rendered_depths = (rendered_end_points_ego_clamped - start_points_ego).norm(dim=-1)
        points_in_volume = (
            # ego volume check
            (end_points_ego >= self.eval_volume_ego.lower.unsqueeze(1)).all(-1)
            & (end_points_ego <= self.eval_volume_ego.upper.unsqueeze(1)).all(-1)
            & (start_points_ego >= self.eval_volume_ego.lower.unsqueeze(1)).all(-1)
            & (start_points_ego <= self.eval_volume_ego.upper.unsqueeze(1)).all(-1)
            # lidar volume check
            & (start_points_lidar >= self.volume.lower.unsqueeze(1)).all(-1)
            & (start_points_lidar <= self.volume.upper.unsqueeze(1)).all(-1)
            & (end_points_lidar >= self.volume.lower.unsqueeze(1)).all(-1)
            & (end_points_lidar <= self.volume.upper.unsqueeze(1)).all(-1)
            # lidar points checks
            & points_mask
            & (lidar_depth > self.min_distance)
        )  # [B, X, Y, Z]

        finite_depth = finite_depth & points_mask & (lidar_depth > self.min_distance)

        out_dict = {
            "rendered_depth": rendered_depths,
            "num_transmission": num_transmissions,
            "lidar_depth": lidar_depth,
            "points_in_ego_volume": points_in_volume,
            "finite_depth": finite_depth,
        }

        return out_dict


METHODS = Literal["cvpr2023", "preds", "bba", "bba04", "open_occupancy"]


class DeltaAccuracy(BinaryAccuracy):
    """Delta accuracy."""

    def __init__(
        self,
        delta: float = 1.25,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        """Delta accuracy."""
        super().__init__(threshold, multidim_average, ignore_index, validate_args, **kwargs)
        self.delta = delta

    def update(self, pred_depth: Tensor, gt_depth: Tensor):
        """Update."""
        thresh = torch.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        binary = thresh < self.delta
        super().update(binary, torch.ones_like(binary))


class Rmse(MeanSquaredError):
    """Root Mean Square Error."""

    def __init__(self, squared: bool = False, num_outputs: int = 1, **kwargs: Any) -> None:
        """Update."""
        super().__init__(squared, num_outputs, **kwargs)


class LogRmse(Rmse):
    """Log Root Mean Square Error."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update."""
        return super().update(preds.log(), target.log())


class RelAbs(MeanAbsoluteError):
    """Relative Absolute Error."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update."""
        return super().update(preds / target, target / target)


class LidarDistanceEval:
    """Lidar distance evaluation."""

    def __init__(
        self,
        ds: NuscenesDataset,
        method: METHODS,
        split: Literal["train", "val"] = "val",
        lower: Sequence[float] = (-40.0, -40.0, -1.0),
        upper: Sequence[float] = (40.0, 40.0, 5.4),
        volume_frame: Literal["ego", "lidar"] = "ego",
        min_distance: float = 2.5,
        batch_size: int = 1,
        p_fn: float = 0.8,
        p_fp: float = 0.05,
        eval_ego_lower: Sequence[float] = (-40.0, -40.0, -1.0),
        eval_ego_upper: Sequence[float] = (40.0, 40.0, 5.4),
        save_path: Optional[str] = None,
        pred_method: Optional[str] = None,
        occ_threshold: Optional[int] = None,
    ) -> None:
        """Lidar distance evaluation."""
        self.ds = ds
        self.split = split
        self.volume = Volume.new_volume(lower=lower, upper=upper).cuda()
        self.eval_volume_ego = Volume.new_volume(lower=eval_ego_lower, upper=eval_ego_upper).cuda()
        self.volume_frame = volume_frame
        self.render_lidar_depth = RenderLidarDepth(
            self.volume,
            min_distance=min_distance,
            volume_frame=self.volume_frame,
            eval_volume_ego=self.eval_volume_ego,
        ).cuda()
        self.method: METHODS = method
        self.save_path = save_path
        self.pred_method = pred_method
        self.occ_threshold = occ_threshold

        metrics = MetricCollection(
            {
                "rmse": Rmse(squared=False),
                "log_rmse": LogRmse(),
                "mae": MeanAbsoluteError(),
                "delta_1_25": DeltaAccuracy(1.25),
                "delta_1_25_2": DeltaAccuracy(1.25**2),
                "delta_1_25_3": DeltaAccuracy(1.25**3),
                "rel_abs": RelAbs(),
            }
        )
        self.metrics = metrics.to(self.volume.device)
        self.batch_size = batch_size

        self.p_fn = p_fn
        self.p_fp = p_fp

    def save_results(self, results: dict):
        """Save results to json."""
        if self.save_path is not None:
            time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            filename = Path(self.save_path) / f"results-{time_str}.json"
            filename.parent.mkdir(exist_ok=True, parents=True)
            results_df = pl.DataFrame(results)
            results_df.write_json(filename)

    def eval(self):
        """Evaluation of dataset."""
        scenes = self.ds.scenes_by_split(self.split)
        for scene in tqdm.tqdm(scenes.iter_slices(1), total=len(scenes)):
            scene = self.ds.join(scene, self.ds.sample)
            self.process_scene(scene)
        results = {
            "method": self.method,
            "version": self.ds.version,
            "split": self.split,
            "occ_threshold": self.occ_threshold,
        }
        if self.method in ["bba", "bba04"]:
            results["p_fn"] = self.p_fn
            results["p_fp"] = self.p_fp
        metrics = self.metrics.compute()
        metrics = {k: v.item() for k, v in metrics.items()}
        results.update(metrics)
        self.save_results(results)
        return results

    def process_scene(self, scene: pl.DataFrame):
        """Process single scene."""
        for sample in scene.iter_slices(self.batch_size):
            sample = self.ds.load_sample_data(sample, "LIDAR_TOP")
            if self.method == "cvpr2023":
                sample = self.ds.load_cvpr2023_occupancy(sample)
            elif self.method == "bba" or self.method == "bba04":
                sample = self.ds.load_reflection_and_transmission_multi_frame(sample)
            elif self.method == "bba" or self.method == "bba04":
                sample = self.ds.load_reflection_and_transmission_multi_frame(sample)
            elif self.method == "open_occupancy":
                sample = self.ds.load_open_occpancy(sample)
            elif self.method == "surround_occ":
                sample = self.ds.load_surround_occ_occupancy(sample)
            elif self.method == "scene_as_occupancy":
                sample = self.ds.load_scene_as_occupancy_gt(sample)
            elif self.method == "preds":
                assert self.pred_method is not None
                sample = self.ds.load_occupancy_preds(sample, self.pred_method)
            self.process_sample(sample)

    def process_sample(self, sample: pl.DataFrame):
        """Process sample."""
        scene_dict = self.load_sample(sample)
        rendered_depth_dict = self.render_lidar_depth(
            occupancy=scene_dict["occupancy"],
            points_lidar=scene_dict["points_lidar"],
            points_mask=scene_dict["points_mask"],
            ego_from_lidar=scene_dict["ego_from_lidar"],
        )
        mask = rendered_depth_dict["points_in_ego_volume"]
        rendered_depth: Tensor = rendered_depth_dict["rendered_depth"][mask]
        lidar_depth: Tensor = rendered_depth_dict["lidar_depth"][mask]
        assert rendered_depth.isfinite().all(), "rendered depth not finite"
        assert lidar_depth.isfinite().all(), "lidar depth not finite"

        self.metrics.update(rendered_depth, lidar_depth)

    def load_sample(self, sample: pl.DataFrame):
        """Load sample."""
        mapping = {
            "points_lidar": "LIDAR_TOP.sample_data.points_lidar",
            "points_mask": "LIDAR_TOP.sample_data.points_mask",
            "ego_from_lidar": "LIDAR_TOP.transform.ego_from_sensor",
        }
        if self.method == "cvpr2023":
            mapping["occupancy"] = "sample.occ_gt.semantics"
        elif self.method == "preds":
            mapping["occupancy"] = f"sample.{self.pred_method}.arr_0"
        elif self.method == "open_occupancy" or self.method == "open_occupancy04":
            mapping["occupancy"] = "sample.open_occupancy.semantics"
        elif self.method == "surround_occ":
            mapping["occupancy"] = "sample.surround_occ.semantics"
        elif self.method == "scene_as_occupancy":
            mapping["occupancy"] = "sample.scene_as_occupancy.semantics"
        elif self.method == "bba" or self.method == "bba04":
            mapping["reflection_and_transmission"] = "LIDAR_TOP.reflection_and_transmission_multi_frame"
        scene_dict = scene_to_tensor_dict(sample, mapping=mapping, device=self.volume.device)
        if self.occ_threshold is not None:
            scene_dict["occupancy"] = torch.where(
                scene_dict["occupancy"] >= self.occ_threshold, 0, 17
            )  # 17: free, otherwise occ
        if self.method == "bba":
            bba = belief_from_reflection_and_transmission_stacked(
                scene_dict["reflection_and_transmission"], p_fn=self.p_fn, p_fp=self.p_fp
            )
            scene_dict["occupancy"] = torch.where(bba[:, 0] > bba[:, 1], 0, 17)
        if self.method == "bba04":
            rt = einops.reduce(
                scene_dict["reflection_and_transmission"], "b c (x 2) (y 2) (z 2) -> b c x y z", reduction="sum"
            )
            bba04 = belief_from_reflection_and_transmission_stacked(rt, p_fn=self.p_fn, p_fp=self.p_fp)
            scene_dict["occupancy"] = torch.where(bba04[:, 0] > bba04[:, 1], 0, 17)
        return scene_dict
