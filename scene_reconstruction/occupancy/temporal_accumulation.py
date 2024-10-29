"""Accumulation of BBA."""
import torch
from torch import Tensor

from scene_reconstruction.core import einsum_transform
from scene_reconstruction.core.volume import Volume
from scene_reconstruction.math.dempster_shafer import yager_rule_of_combination_stacked


def recursive_accumulation(m: Tensor, volume: Volume, ego_from_global: torch.Tensor, global_from_ego: torch.Tensor):
    """Recursive yager rule."""
    volume = volume.cuda()
    grid_current_frame = volume.new_coord_grid(volume.volume_shape(m))
    m_out = torch.empty_like(m)
    m_out[0] = m[0]
    last_from_current = (ego_from_global[:-1] @ global_from_ego[1:]).cuda()
    for i in range(1, len(m)):
        grid_last_frame = einsum_transform("blc,bxyzc->bxyzl", last_from_current[i - 1 : i], points=grid_current_frame)
        m_sampled = volume.sample_volume(m_out[i - 1 : i].cuda(), grid_last_frame.cuda())
        m_out[i : i + 1] = yager_rule_of_combination_stacked(m_sampled, m[i : i + 1].cuda()).cpu()
    return m_out


def recursive_accumulation_backward(m, volume: Volume, ego_from_global: torch.Tensor, global_from_ego: torch.Tensor):
    """Recursive yager rule backwards."""
    volume = volume.cuda()
    grid_current_frame = volume.new_coord_grid(volume.volume_shape(m))
    m_out = torch.empty_like(m)
    m_out[-1] = m[-1]
    next_from_curr = (ego_from_global[1:] @ global_from_ego[:-1]).cuda()
    for i in range(len(m) - 1, 0, -1):
        grid_next_frame = einsum_transform("blc,bxyzc->bxyzl", next_from_curr[i - 1 : i], points=grid_current_frame)
        m_sampled = volume.sample_volume(m_out[i : i + 1].cuda(), grid_next_frame.cuda())
        m_out[i - 1 : i] = yager_rule_of_combination_stacked(m_sampled, m[i - 1 : i].cuda()).cpu()
    return m_out


def forward_backward_accumulation(
    m: Tensor, volume: Volume, ego_from_global: torch.Tensor, global_from_ego: torch.Tensor
):
    """Recursive yager rule forward and backwards."""
    m_forward = recursive_accumulation(m, volume, ego_from_global=ego_from_global, global_from_ego=global_from_ego)
    m_backward = recursive_accumulation_backward(
        m, volume, ego_from_global=ego_from_global, global_from_ego=global_from_ego
    )
    m_forward_backward = yager_rule_of_combination_stacked(m_forward, m_backward)
    return m_forward_backward
