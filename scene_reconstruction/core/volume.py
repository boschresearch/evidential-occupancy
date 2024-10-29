"""Dense 3D Volumes."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .transform import einsum_transform, transform_to_grid_sample_coords, transform_volume_bounds


@dataclass
class Volume:
    """Indexing for 3D volume of shape [B, C, X, Y, Z]."""

    lower: Tensor  # [B, 3]
    upper: Tensor  # [B, 3]

    @classmethod
    def new_volume(cls, lower: Sequence[float], upper: Sequence[float], **kwargs):
        """New volume from lower and upper bound."""
        assert len(lower) == 3 and len(upper) == 3
        return cls(
            lower=torch.tensor([lower], **kwargs),
            upper=torch.tensor([upper], **kwargs),
        )

    @classmethod
    def new_normalized(cls, **kwargs):
        """New volume from -1.0, to 1.0."""
        return cls.new_volume(lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 1.0], **kwargs)

    @classmethod
    def new_index(cls, volume_shape: Sequence[int], **kwargs):
        """New volume from 0 to volume_shape."""
        return cls.new_volume([0, 0, 0], volume_shape, **kwargs)

    @staticmethod
    def volume_shape(features: Tensor):
        """Volume shape of features."""
        return features.shape[-3:]

    @property
    def device(self) -> torch.device:
        """Device of data."""
        assert self.lower.device == self.upper.device
        return self.lower.device

    def to(self, *args, **kwargs) -> "Volume":
        """Convert dtype and/or device."""
        return Volume(lower=self.lower.to(*args, **kwargs), upper=self.upper.to(*args, **kwargs))

    def cuda(self, *args, **kwargs) -> "Volume":
        """Move to cuda device."""
        return Volume(lower=self.lower.cuda(*args, **kwargs), upper=self.upper.cuda(*args, **kwargs))

    def voxel_size(self, features: Tensor) -> Tensor:
        """Voxel size."""
        return (self.upper - self.lower).abs() / torch.tensor(self.volume_shape(features), device=self.device)

    def voxel_size_from_shape(self, volume_shape: Sequence[int]) -> Tensor:
        """Voxel size for given volume shape."""
        return (self.upper - self.lower).abs() / torch.tensor(volume_shape, device=self.device)

    def other_from_self(self, other: "Volume") -> Tensor:
        """Homogenous transformation from own volume to other volume."""
        return transform_volume_bounds(self.lower, self.upper, other.lower, other.upper)

    def self_from_other(self, other: "Volume") -> Tensor:
        """Homogenous transformation from other volume to own volume."""
        return other.other_from_self(self)

    def _grid_sample_from_self(self) -> Tensor:
        """Homogenous transformation to grid_sample compatible coordinates."""
        return transform_to_grid_sample_coords(self.lower, self.upper)

    def new_coord_grid(self, volume_shape: Sequence[int]) -> Tensor:
        """Uniform grid coordinates over volume."""
        # coordinates correspond to centers of each voxel
        assert len(volume_shape) == 3
        coordinates = [(torch.arange(num_steps, device=self.device) + 0.5) / num_steps for num_steps in volume_shape]
        grid = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), -1)
        grid = (self.upper - self.lower)[:, None, None, None, :] * grid + self.lower[:, None, None, None, :]
        return grid

    def coord_grid(self, features: Tensor, expand: bool = True) -> Tensor:
        """Grid coordinates for the given features."""
        # coordinates correspond to centers of each voxel
        return self.new_coord_grid(self.volume_shape(features)).expand(
            features.shape[0] if expand else 1, -1, -1, -1, -1
        )

    def sample_volume(
        self, volume_features: Tensor, sample_coords: Tensor, fill_invalid: float = 0.0, mode: str = "bilinear"
    ) -> Tensor:
        """Samples the provided features at given sample points."""
        # volume_features: [B, C, X, Y, Z]
        # sample_coords: [B, X, Y, Z, 3]
        normalized_from_grid = self._grid_sample_from_self()
        normalized_coords = einsum_transform("bng,bxyzg->bxyzn", normalized_from_grid, points=sample_coords)
        normalized_coords = normalized_coords.expand(volume_features.shape[0], -1, -1, -1, -1)
        return torch.nn.functional.grid_sample(
            volume_features, normalized_coords, mode=mode, align_corners=False
        ).nan_to_num(fill_invalid)

    def clamp_points_along_line(self, points: Tensor, line_origin: Tensor):
        """Clamps points to volume bounds along given ray direction."""
        # points [B, N, 3]
        # line_origin [B, N, 3]
        direction = points - line_origin
        direction = direction / direction.norm(dim=-1, keepdim=True)
        below_lower = (points - self.lower.unsqueeze(1)).clamp_max(0)
        above_upper = (points - self.upper.unsqueeze(1)).clamp_min(0)
        s = torch.maximum(
            (below_lower / direction).nan_to_num(0).amax(-1, keepdim=True), (above_upper / direction).nan_to_num(0).amax(-1, keepdim=True)
        )
        boxed = points - s * direction
        return boxed
