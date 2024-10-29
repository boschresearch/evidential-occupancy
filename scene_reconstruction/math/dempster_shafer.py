"""Dempster Shafer theory."""
import itertools
import math

import einops
import torch
from torch import Tensor


def yager_rule_of_combination_stacked(m1: Tensor, m2: Tensor, dim=1):
    """Yager's rule of combination with stacked BBA."""
    m_omega1 = 1.0 - m1.sum(dim, keepdim=True)
    m_omega2 = 1.0 - m2.sum(dim, keepdim=True)
    m = m1 * m2 + m1 * m_omega2 + m_omega1 * m2
    return m


def yager_rule_of_combination_single_element(m_of: Tensor):
    """Yager's rule of combination with stacked BBA."""
    # m_of: 2 B
    m_o, m_f = m_of.unbind(0)  # B
    m_omega = 1.0 - m_o - m_f
    m_o = torch.cartesian_prod(*torch.stack([m_o, m_omega]).unbind(1)).prod(-1).sum() - m_omega.prod()
    m_f = torch.cartesian_prod(*torch.stack([m_f, m_omega]).unbind(1)).prod(-1).sum() - m_omega.prod()
    m_of = torch.stack([m_o, m_f], -1)
    return m_of


def yager_rule_of_combination_across_batch(m: Tensor):
    """Yager's rule of combination over first dimension with stacked BBA."""
    # [B, 2, X, Y, Z]
    _, _, X, Y, Z = m.shape
    m_flat = einops.rearrange(m, "b m x y z -> (x y z) m b")
    m_combined_flat = torch.vmap(yager_rule_of_combination_single_element)(m_flat)
    m_combined = einops.repeat(m_combined_flat, "(x y z) m -> 1 m x y z", x=X, y=Y, z=Z)
    return m_combined


def yager_rule_of_combination_across_batch_iterative(m: Tensor, channel_dim=1, combination_dim: int = 0):
    """Yager's rule of combination over first dimension with stacked BBA."""
    # [B, 2, X, Y, Z]
    if m.shape[combination_dim] == 1:
        return m
    omega = 1.0 - m.sum(channel_dim, keepdim=True)
    out = torch.zeros_like(m.select(combination_dim, 0))  # [2, X, Y, Z]

    # iterate over all combinations of m and omega
    for first, *combination in itertools.product(*zip(m.unbind(combination_dim), omega.unbind(combination_dim))):
        # and add them up
        out += math.prod(combination, start=first)
    # correct for all omega combination
    omega_first, *omega_other = omega.unbind(combination_dim)
    out -= math.prod(omega_other, start=omega_first)
    return out


def dempster_rule_of_combination(m_o1, m_o2, m_f1, m_f2):
    """Dempster's rule of combination."""
    m_omega1 = 1 - (m_o1 + m_f1)
    m_omega2 = 1 - (m_o2 + m_f2)
    conflicts = m_o1 * m_f2 + m_f1 * m_o2
    m_o = (m_o1 * m_o2 + m_o1 * m_omega2 + m_omega1 * m_o2) / (1 - conflicts).clamp_min(1e-8)
    m_f = (m_f1 * m_f2 + m_f1 * m_omega2 + m_omega1 * m_f2) / (1 - conflicts).clamp_min(1e-8)
    return m_o, m_f


def yager_rule_of_combination(m_o1, m_o2, m_f1, m_f2):
    """Yager's rule of combination."""
    m_omega1 = 1 - (m_o1 + m_f1)
    m_omega2 = 1 - (m_o2 + m_f2)
    m_o = m_o1 * m_o2 + m_o1 * m_omega2 + m_omega1 * m_o2
    m_f = m_f1 * m_f2 + m_f1 * m_omega2 + m_omega1 * m_f2
    return m_o, m_f


def belief_from_reflection_and_transmission(num_reflections, num_transmissions, p_fn=0.8, p_fp=0.1):
    """Basic belief assignement from number of reflections and transmissions."""
    # p_fn = 0.9  # occupied with transmission
    # p_fp = 0.6  # empty with reflection
    # https://arxiv.org/pdf/1801.05297.pdf

    m_o = p_fn**num_transmissions * (1.0 - p_fp**num_reflections)
    m_f = p_fp**num_reflections * (1.0 - p_fn**num_transmissions)
    return torch.cat([m_o, m_f], 1)


def belief_from_reflection_and_transmission_stacked(num_rt: Tensor, p_fn=0.8, p_fp=0.05, with_omega: bool = False):
    """Basic belief assignement from number of reflections and transmissions."""
    # p_fn = 0.9  # occupied with transmission
    # p_fp = 0.6  # empty with reflection
    # https://arxiv.org/pdf/1801.05297.pdf
    num_reflections, num_transmissions = num_rt.split(1, 1)
    m_o = p_fn**num_transmissions * (1.0 - p_fp**num_reflections)
    m_f = p_fp**num_reflections * (1.0 - p_fn**num_transmissions)
    if with_omega:
        m_omega = 1.0 - m_o - m_f
        return torch.cat([m_o, m_f, m_omega], 1)
    return torch.cat([m_o, m_f], 1)
