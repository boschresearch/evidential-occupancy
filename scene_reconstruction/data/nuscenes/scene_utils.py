"""Scene utils."""
import polars as pl

from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch


def scene_to_tensor_dict(scene: pl.DataFrame, mapping: dict[str, str], **kwargs):
    """Dataframe to dict of tensors."""
    return {k: series_to_torch(scene[v]).to(**kwargs) for k, v in mapping.items()}
