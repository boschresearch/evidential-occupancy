"""Polars utils."""

import re
from typing import Literal, Optional

import numpy as np
import polars as pl
import pyarrow as pa
import torch
from polars.type_aliases import JoinStrategy, JoinValidation
from scipy.spatial.transform import Rotation

# pylint: disable=C0103


def numpy_to_arrow(x: np.ndarray):
    """Numpy array into a a nested FixedSizeListArray."""
    shape = x.shape
    arr = pa.array(x.ravel())
    for last_dim in shape[-1:0:-1]:
        arr = pa.FixedSizeListArray.from_arrays(arr, last_dim)
    return arr


def numpy_to_series(name: str, values: np.ndarray):
    """Numpy array to series."""
    return pl.Series(name=name, values=numpy_to_arrow(values))


def torch_to_series(name: str, values: torch.Tensor):
    """Torch tensor to series."""
    return numpy_to_series(name, values.numpy())


def series_to_numpy(
    series: pl.Series,
    zero_copy_only: bool = False,
    writable: bool = False,
    use_pyarrow: bool = True,
):
    """Polars series to numpy array."""
    shape = [len(series)]
    while series.dtype == pl.Array:
        shape.append(series.dtype.width)
        series = series.explode()
    return series.to_numpy(zero_copy_only=zero_copy_only, writable=writable, use_pyarrow=use_pyarrow).reshape(shape)


def series_to_torch(
    series: pl.Series,
    zero_copy_only: bool = False,
    use_pyarrow: bool = True,
):
    """Polars series to torch array."""
    return torch.from_numpy(
        series_to_numpy(series=series, zero_copy_only=zero_copy_only, writable=True, use_pyarrow=use_pyarrow)
    )


TOKEN_REGEX = r"^.*\.token$"


def col_is_token(col: str):
    """Check if column matches token regex."""
    return re.search(TOKEN_REGEX, col, re.DOTALL) is not None


def common_tokens(*dfs: pl.DataFrame):
    """Common tokens of multiple DataFrames."""
    assert dfs
    all_tokens = [set(filter(col_is_token, df.columns)) for df in dfs]
    common = set.intersection(*all_tokens)
    return sorted(list(common))


def common_non_tokens(*dfs: pl.DataFrame):
    """Common non token columns of multiple DataFrames."""
    assert dfs, "No DataFrames are given."
    all_non_tokens = [set(filter(lambda x: not col_is_token(x), df.columns)) for df in dfs]
    common = set.intersection(*all_non_tokens)
    return sorted(list(common))


def join_on_token(
    df: pl.DataFrame,
    other: pl.DataFrame,
    how: JoinStrategy = "inner",
    *,
    suffix: str = "_right",
    allow_duplicates: bool = False,
    validate: JoinValidation = "m:m",
    verbose: bool = False,
):
    """Join DataFrames on based on tokens."""
    tokens = common_tokens(df, other)
    assert (
        allow_duplicates or len(common_non_tokens(df, other)) == 0
    ), f"There are duplicate columns:  {common_non_tokens(df, other)}"
    if verbose:
        print(f"Joining on {tokens}")
    return df.join(other, on=tokens, how=how, suffix=suffix, validate=validate)


def quaternion_to_matrix(quat: np.ndarray, quat_order: Literal["wxyz", "xyzw"] = "wxyz"):
    """Convert quaternion to rotation matrix."""
    if quat_order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix().astype(quat.dtype)
    return rot_mat


def homogenous_transform(*, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
    """Build homgenous transform from rotation and translation."""
    *batch_size_translation, _3 = translation_vector.shape
    *batch_size_rotation, _3, _3 = rotation_matrix.shape
    assert batch_size_translation == batch_size_rotation
    homo = np.zeros(batch_size_rotation + [4, 4], dtype=rotation_matrix.dtype)
    homo[:, :3, :3] = rotation_matrix
    homo[:, :3, 3] = translation_vector
    homo[:, 3, 3] = 1.0
    return homo


def inverse_transform(transform: np.ndarray):
    """Inverse homogenous transform."""
    transform_inv = np.empty_like(transform)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_inv = np.einsum("...ij->...ji", rot)
    trans_inv = -np.einsum("...ji,...i->...j", rot_inv, trans)
    transform_inv[..., :3, :3] = rot_inv
    transform_inv[..., :3, 3] = trans_inv
    transform_inv[..., 3, :3] = 0
    transform_inv[..., 3, 3] = 1
    return transform_inv


def transform_from_columns(
    df: pl.DataFrame,
    rotation: str,
    translation: str,
    *,
    transform: Optional[str] = None,
    transform_inv: Optional[str] = None,
):
    """Parse homogenous transform from columns."""
    if transform is None and transform_inv is None:
        return df
    rotation_quat = series_to_numpy(df[rotation])
    translation_vec = series_to_numpy(df[translation])
    rotation_matrix = quaternion_to_matrix(rotation_quat)
    transform_matrix = homogenous_transform(rotation_matrix=rotation_matrix, translation_vector=translation_vec)
    new_cols = {}
    if transform is not None:
        new_cols[transform] = transform_matrix
    if transform_inv is not None:
        new_cols[transform_inv] = inverse_transform(transform_matrix)
    return df.with_columns(*[numpy_to_series(k, v) for k, v in new_cols.items()])


def pad_intrinsics_to_4x4(intrinsics: np.ndarray):
    """Pad intrinsics to homogenous transform."""
    *batch_size_rotation, _3, _3 = intrinsics.shape
    homo = np.zeros(batch_size_rotation + [4, 4], dtype=intrinsics.dtype)
    homo[:, :3, :3] = intrinsics
    homo[:, 3, 3] = 1.0
    return homo


def pad_intrinsics_from_colums(df: pl.DataFrame, intrinsics: str, transform: str):
    """Pad intrinsics columns to homogenous transform."""
    matrix = series_to_numpy(df[intrinsics])
    transform_np = pad_intrinsics_to_4x4(matrix)
    return df.with_columns(numpy_to_series(transform, transform_np))


def nested_to_numpy(x, fill_value=None):
    """Nested numpy arrays to single numpy tensor."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        to_stack = [nested_to_numpy(i, fill_value=fill_value) if i is not None else fill_value for i in x]
        return np.stack(to_stack)
    return x


def nested_to_torch(x, fill_value=None):
    """Nested numpy arrays to torch tensor."""
    return torch.from_numpy(nested_to_numpy(x, fill_value=fill_value))
