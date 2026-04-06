#  Copyright 2025 ETH Zurich
#  Created by OpenAI Codex, adapted from phasor reference implementation
#  SPDX-License-Identifier: BSD-3-Clause

"""Phasor backbone utilities for navigation policies."""

from __future__ import annotations

import torch
import torch.nn as nn

PFL_PHASE_SHIFTS_DEG = (-90.0, 90.0, -180.0)


def _phase_shift_index(signal_size: int, degrees: float) -> int:
    """Convert a phase shift in degrees into a circular index offset."""
    return int(round(signal_size * (degrees / 360.0))) % signal_size


def _validate_circular_corr_inputs(source: torch.Tensor, target: torch.Tensor) -> int:
    """Validate circular-correlation inputs and return the shared signal size."""
    if source.shape != target.shape:
        raise ValueError("circular correlation requires source and target to have identical shapes")
    if source.ndim < 1:
        raise ValueError("circular correlation requires at least one dimension")
    return int(source.shape[-1])


def circular_corr_fft(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute circular correlation over the last dimension via FFT."""
    signal_size = _validate_circular_corr_inputs(source, target)
    source_freq = torch.fft.rfft(source, dim=-1)
    target_freq = torch.fft.rfft(target, dim=-1)
    return torch.fft.irfft(source_freq * torch.conj(target_freq), n=signal_size, dim=-1) / signal_size


class DepthVerticalPool(nn.Module):
    """Pool each depth column independently into one ring-aligned feature."""

    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.proj = nn.Linear(self.image_height, 1)

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Pool a ``(B, H, W)`` raw depth image into ``(B, W)`` features."""
        depth_image = depth_image.reshape(-1, self.image_height, self.image_width)
        depth_image = depth_image.transpose(-2, -1)
        return self.proj(depth_image).squeeze(-1)


class NavigationPhasorBackbone(nn.Module):
    """Convert navigation state vectors plus visual features into phasor inputs."""

    def __init__(
        self,
        num_observations: int,
        ring_size: int = 64,
        depth_image_height: int = 40,
        depth_image_width: int = 64,
        state_obs_dim: int = 8,
        visual_feature_dim: int = 64,
    ):
        super().__init__()

        self.num_observations = int(num_observations)
        self.ring_size = int(ring_size)
        self.state_obs_dim = int(state_obs_dim)
        self.visual_feature_dim = int(visual_feature_dim)
        self.depth_image_height = int(depth_image_height)
        self.depth_image_width = int(depth_image_width)
        self.num_depth_features = self.depth_image_height * self.depth_image_width
        self.expected_obs_dim = self.state_obs_dim + self.visual_feature_dim
        if self.num_observations != self.expected_obs_dim:
            raise ValueError(
                f"NavigationPhasorBackbone expected {self.expected_obs_dim} observations, "
                f"got {self.num_observations}"
            )
        if self.visual_feature_dim != self.ring_size:
            raise ValueError("visual_feature_dim must match ring size for the phasor SRU input layout")

        neuron_angles = torch.linspace(0.0, 2.0 * torch.pi, self.ring_size + 1)[:-1]
        neuron_pos = torch.stack((torch.cos(neuron_angles), torch.sin(neuron_angles)), dim=1)
        pfl_shift_indices = torch.tensor(
            [_phase_shift_index(self.ring_size, shift) for shift in PFL_PHASE_SHIFTS_DEG],
            dtype=torch.long,
        )

        # Keep the raw-depth pooling path around for future ablations, but the
        # active phasor stack now uses the same VAE-encoded + cross-attended
        # depth frontend as the baseline SRU policies.
        # self.depth_pool = DepthVerticalPool(self.depth_image_height, self.depth_image_width)
        self.register_buffer("neuron_pos", neuron_pos)
        self.register_buffer("pfl_shift_indices", pfl_shift_indices)

    def pos2phasor(self, pos: torch.Tensor) -> torch.Tensor:
        """Project a ``(..., 2)`` direction vector onto the ring basis."""
        return torch.matmul(pos, self.neuron_pos.transpose(0, 1))

    def encode_from_state_obs(
        self,
        state_obs: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode flat state observations plus fused visual features into phasor inputs."""
        if state_obs.shape[-1] != self.state_obs_dim:
            raise ValueError(f"Expected state_obs dim {self.state_obs_dim}, got {state_obs.shape[-1]}")
        if visual_features.shape[-1] != self.visual_feature_dim:
            raise ValueError(f"Expected visual_features dim {self.visual_feature_dim}, got {visual_features.shape[-1]}")

        forward_vel = state_obs[..., 0:1]
        yaw_rate = state_obs[..., 1:2]
        heading_trig = state_obs[..., 2:4]
        goal_pos_error = state_obs[..., 4:6]
        goal_heading_trig = state_obs[..., 6:8]

        depth_1d = visual_features
        # raw_depth = observations[..., 8:].reshape(-1, self.depth_image_height, self.depth_image_width)
        # depth_1d = self.depth_pool(raw_depth)
        phasor_heading = self.pos2phasor(heading_trig)
        phasor_pos_error = self.pos2phasor(goal_pos_error)
        phasor_goal_heading = self.pos2phasor(goal_heading_trig)
        phasor_hdg_error = phasor_goal_heading - phasor_heading
        sru_input = torch.cat(
            (depth_1d, phasor_heading, phasor_pos_error, phasor_hdg_error, forward_vel, yaw_rate),
            dim=-1,
        )
        return forward_vel, yaw_rate, phasor_heading, phasor_pos_error, phasor_hdg_error, depth_1d, sru_input

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode already-fused phasor observations.

        ``observations`` must contain the 8 state scalars followed by the
        64-D visual feature vector produced by the shared cross-attention depth
        frontend. The legacy raw-depth vertical-pooling path is intentionally
        disabled while we match the baseline vision encoder.
        """
        state_obs = observations[..., : self.state_obs_dim]
        visual_features = observations[..., self.state_obs_dim :]
        return self.encode_from_state_obs(state_obs, visual_features)


__all__ = [
    "PFL_PHASE_SHIFTS_DEG",
    "DepthVerticalPool",
    "NavigationPhasorBackbone",
    "circular_corr_fft",
]
