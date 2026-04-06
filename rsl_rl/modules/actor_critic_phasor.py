#  Copyright 2025 ETH Zurich
#  Created by OpenAI Codex, adapted from phasor reference implementation
#  SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic module with a phasor backbone and SRU memory."""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks.phasor_backbone import NavigationPhasorBackbone
from rsl_rl.networks.sru_memory import CrossAttentionFuseModule
from rsl_rl.modules.actor_critic_recurrent import Memory
from rsl_rl.modules.actor_critic_sru import LinearConstDropout, get_activation
from rsl_rl.utils import unpad_trajectories


def _build_mlp(output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential | nn.Linear:
    """Build the trailing MLP after the optional linear+dropout input block."""
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must be non-empty when building an MLP head")

    layers: list[nn.Module] = []
    for layer_index, hidden_dim in enumerate(hidden_dims):
        if layer_index == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dims[layer_index + 1]))
            layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class ActorCriticPhasor(nn.Module):
    """Actor-critic with phasor inputs and a standard recurrent MLP policy head."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Optional[list[int]] = None,
        critic_hidden_dims: Optional[list[int]] = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        image_input_dims: tuple[int, int, int] = (64, 5, 8),
        height_input_dims: tuple[int, int, int] = (64, 7, 7),
        rnn_type: str = "lstm_sru",
        dropout: float = 0.0,
        rnn_hidden_size: int = 64,
        rnn_num_layers: int = 1,
        num_cameras: int = 1,
        **kwargs,
    ):
        if actor_hidden_dims is None:
            actor_hidden_dims = []
        if critic_hidden_dims is None:
            critic_hidden_dims = [128, 64]

        phasor_N = int(kwargs.pop("phasor_N", image_input_dims[0]))
        time_embed_dim = int(kwargs.pop("time_embed_dim", 8))
        depth_image_height = int(kwargs.pop("depth_image_height", image_input_dims[1]))
        depth_image_width = int(kwargs.pop("depth_image_width", image_input_dims[2]))
        if kwargs:
            print(f"[ActorCriticPhasor] Warning: got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        if num_cameras != 1:
            raise ValueError("ActorCriticPhasor currently supports a single encoded depth camera only")
        if image_input_dims[0] != phasor_N:
            raise ValueError("phasor_N must match the encoded visual feature width")
        if rnn_type not in ("lstm", "gru", "lstm_sru", "lstm_a_gate"):
            raise ValueError("ActorCriticPhasor supports rnn_type in {'lstm', 'gru', 'lstm_sru'}")

        super().__init__()

        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.image_input_dims = image_input_dims
        self.height_input_dims = height_input_dims
        self.ring_size = phasor_N
        self.rnn_type = rnn_type
        self.time_embed_dim = time_embed_dim
        self.depth_image_height = depth_image_height
        self.depth_image_width = depth_image_width
        self.num_image_features = image_input_dims[0] * image_input_dims[1] * image_input_dims[2]
        self.num_height_features = height_input_dims[0] * height_input_dims[1] * height_input_dims[2]
        self.actor_core_state_obs_dim = 8
        self.actor_total_state_obs_dim = self.num_actor_obs - self.num_image_features * num_cameras
        self.actor_aux_scalar_dim = self.actor_total_state_obs_dim - self.actor_core_state_obs_dim
        if self.actor_aux_scalar_dim != 8:
            raise ValueError(
                "ActorCriticPhasor expects actor observations to include base_lin_vel(3), "
                "base_ang_vel(3), and last_action(2) after the 8 phasor state dims"
            )
        if self.actor_core_state_obs_dim != 8:
            raise ValueError(
                "ActorCriticPhasor expects 8 non-visual state observations: "
                "[forward_vel, yaw_rate, heading(2), goal_pos_error(2), goal_heading(2)]"
            )
        self.critic_state_obs_dim = self.actor_core_state_obs_dim
        self.critic_privileged_scalar_dim = (
            self.num_critic_obs
            - self.num_image_features * num_cameras
            - self.num_height_features
            - 1
            - self.critic_state_obs_dim
        )
        if self.critic_privileged_scalar_dim < 0:
            raise ValueError("Critic observation layout is too small for phasor critic parsing")
        self.critic_attention_info_dim = self.critic_state_obs_dim + self.critic_privileged_scalar_dim
        self.sru_input_dim = 4 * self.ring_size + 2

        self.attn_image_net = CrossAttentionFuseModule(
            image_dim=image_input_dims[0],
            info_dim=self.actor_total_state_obs_dim,
            num_heads=4,
            spatial_dims=(num_cameras, image_input_dims[1], image_input_dims[2]),
        )
        self.attn_critic_image_net = CrossAttentionFuseModule(
            image_dim=image_input_dims[0],
            info_dim=self.critic_attention_info_dim,
            num_heads=4,
            spatial_dims=(num_cameras, image_input_dims[1], image_input_dims[2]),
        )
        self.attn_height_net = CrossAttentionFuseModule(
            image_dim=height_input_dims[0],
            info_dim=self.critic_attention_info_dim,
            num_heads=4,
            spatial_dims=(1, height_input_dims[1], height_input_dims[2]),
        )
        self.time_layer = nn.Linear(1, self.time_embed_dim)

        self.backbone_a = NavigationPhasorBackbone(
            num_observations=self.actor_core_state_obs_dim + image_input_dims[0],
            ring_size=self.ring_size,
            depth_image_height=self.depth_image_height,
            depth_image_width=self.depth_image_width,
            state_obs_dim=self.actor_core_state_obs_dim,
            visual_feature_dim=image_input_dims[0],
        )
        self.backbone_c = NavigationPhasorBackbone(
            num_observations=self.critic_state_obs_dim + image_input_dims[0],
            ring_size=self.ring_size,
            depth_image_height=self.depth_image_height,
            depth_image_width=self.depth_image_width,
            state_obs_dim=self.critic_state_obs_dim,
            visual_feature_dim=image_input_dims[0],
        )
        self.memory_a = Memory(
            self.sru_input_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )
        self.memory_c = Memory(
            self.sru_input_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )

        self.actor_input_dim = rnn_hidden_size
        self.critic_input_dim = rnn_hidden_size + height_input_dims[0] + self.critic_privileged_scalar_dim + self.time_embed_dim

        if len(actor_hidden_dims) == 0:
            self.linear_dropout_actor = None
            self.actor = nn.Linear(self.actor_input_dim, num_actions)
        else:
            self.linear_dropout_actor = LinearConstDropout(
                self.actor_input_dim,
                actor_hidden_dims[0],
                dropout_p=dropout,
                activation_name=activation,
            )
            self.actor = _build_mlp(num_actions, actor_hidden_dims, activation)

        if len(critic_hidden_dims) == 0:
            self.linear_dropout_critic = None
            self.critic = nn.Linear(self.critic_input_dim, 1)
        else:
            self.linear_dropout_critic = LinearConstDropout(
                self.critic_input_dim,
                critic_hidden_dims[0],
                dropout_p=dropout,
                activation_name=activation,
            )
            self.critic = _build_mlp(1, critic_hidden_dims, activation)

        if isinstance(init_noise_std, (list, tuple)):
            init_std = torch.tensor(init_noise_std, dtype=torch.float32)
        else:
            init_std = float(init_noise_std) * torch.ones(num_actions, dtype=torch.float32)
        self.log_std = nn.Parameter(torch.log(init_std))
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticPhasor] Actor head: {self.actor}")
        print(f"[ActorCriticPhasor] Critic head: {self.critic}")
        print(f"[ActorCriticPhasor] Actor RNN: {self.memory_a}")
        print(f"[ActorCriticPhasor] Critic RNN: {self.memory_c}")
        print(
            "[ActorCriticPhasor] "
            f"ring_size={self.ring_size}, image={self.image_input_dims}, "
            f"actor_obs={self.num_actor_obs}, critic_obs={self.num_critic_obs}, "
            f"rnn_type={self.rnn_type}, "
            f"actor_aux_scalar_dim={self.actor_aux_scalar_dim}, "
            f"critic_privileged_scalar_dim={self.critic_privileged_scalar_dim}"
        )

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_parameters(self):
        params = (
            list(self.attn_image_net.parameters())
            + list(self.backbone_a.parameters())
            + list(self.memory_a.parameters())
            + list(self.actor.parameters())
        )
        if self.linear_dropout_actor is not None:
            params += list(self.linear_dropout_actor.parameters())
        params += [self.log_std]
        return params

    def get_critic_parameters(self):
        params = (
            list(self.attn_height_net.parameters())
            + list(self.attn_critic_image_net.parameters())
            + list(self.backbone_c.parameters())
            + list(self.memory_c.parameters())
            + list(self.time_layer.parameters())
            + list(self.critic.parameters())
        )
        if self.linear_dropout_critic is not None:
            params += list(self.linear_dropout_critic.parameters())
        return params

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError("Forward method not implemented.")

    def _encode_actor_with_backbone(
        self,
        observations: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode actor observations with state-conditioned image attention."""
        batch_mode = masks is not None
        leading_shape = observations.shape[:-1]
        flat_obs = observations.reshape(-1, observations.shape[-1])
        actor_info = flat_obs[..., : self.actor_total_state_obs_dim]
        actor_core_state = actor_info[..., : self.actor_core_state_obs_dim]
        image_obs = flat_obs[..., self.actor_total_state_obs_dim :].reshape(-1, *self.image_input_dims)
        image_features = self.attn_image_net(image_obs, actor_info)
        _, _, _, _, _, _, sru_input = self.backbone_a.encode_from_state_obs(actor_core_state, image_features)

        if batch_mode:
            seq_len, batch_size = leading_shape
            sru_input = sru_input.view(seq_len, batch_size, -1)
        return sru_input

    def _process_actor_core(self, observations, masks=None, hidden_states=None):
        sru_input = self._encode_actor_with_backbone(observations, masks)
        actor_state = self.memory_a(sru_input, masks, hidden_states)
        return actor_state.squeeze(0)

    def _process_critic_core(self, observations, masks=None, hidden_states=None):
        batch_mode = masks is not None
        flat_obs = observations.reshape(-1, observations.shape[-1])
        critic_info_end = self.critic_state_obs_dim + self.critic_privileged_scalar_dim
        time_end = critic_info_end + 1
        height_end = time_end + self.num_height_features

        state_obs = flat_obs[..., : self.critic_state_obs_dim]
        critic_info = flat_obs[..., :critic_info_end]
        privileged_obs = flat_obs[..., self.critic_state_obs_dim:critic_info_end]
        time_obs = observations[..., critic_info_end:time_end]
        height_obs = flat_obs[..., time_end:height_end].reshape(-1, *self.height_input_dims)
        image_obs = flat_obs[..., height_end:].reshape(-1, *self.image_input_dims)

        height_features = self.attn_height_net(height_obs, critic_info)
        image_features = self.attn_critic_image_net(image_obs, critic_info)
        _, _, _, _, _, _, sru_input = self.backbone_c.encode_from_state_obs(state_obs, image_features)
        if batch_mode:
            seq_len, batch_size, _ = observations.shape
            sru_input = sru_input.view(seq_len, batch_size, -1)
        critic_state = self.memory_c(sru_input, masks, hidden_states)
        time_embed = self.time_layer(time_obs)

        if batch_mode:
            height_features = height_features.view(seq_len, batch_size, -1)
            privileged_obs = privileged_obs.view(seq_len, batch_size, -1)
            height_features = unpad_trajectories(height_features, masks)
            privileged_obs = unpad_trajectories(privileged_obs, masks)
            time_embed = unpad_trajectories(time_embed, masks)

        return torch.cat((critic_state.squeeze(0), height_features, privileged_obs, time_embed), dim=-1)

    def _compute_action_mean(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        actor_input = self._process_actor_core(observations, masks, hidden_states)
        if self.linear_dropout_actor is not None:
            actor_input = self.linear_dropout_actor(actor_input, dropout_masks)
        return self.actor(actor_input)

    def update_distribution(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        mean = self._compute_action_mean(observations, masks, hidden_states, dropout_masks)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def _compute_value(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        critic_input = self._process_critic_core(critic_observations, masks, hidden_states)
        if self.linear_dropout_critic is not None:
            critic_input = self.linear_dropout_critic(critic_input, dropout_masks)
        return self.critic(critic_input)

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        self.update_distribution(observations, masks, hidden_states, dropout_masks)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        return self._compute_action_mean(observations, masks, hidden_states, dropout_masks)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        return self._compute_value(critic_observations, masks, hidden_states, dropout_masks)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_dropout_masks(self):
        actor_mask = None if self.linear_dropout_actor is None else self.linear_dropout_actor.get_dropout_mask()
        critic_mask = None if self.linear_dropout_critic is None else self.linear_dropout_critic.get_dropout_mask()
        return actor_mask, critic_mask

    def reset_dropout_masks(self):
        if self.linear_dropout_actor is not None:
            self.linear_dropout_actor.reset_dropout_mask()
        if self.linear_dropout_critic is not None:
            self.linear_dropout_critic.reset_dropout_mask()

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        """Export the actor path as a stateful TorchScript policy."""
        rnn = self.memory_a.rnn
        rnn_type = type(rnn).__name__.lower()
        exporter_kwargs = dict(
            backbone=self.backbone_a,
            attn_image_net=self.attn_image_net,
            rnn=rnn,
            forward_input_layer=self.linear_dropout_actor,
            actor=self.actor,
            image_input_dims=self.image_input_dims,
            num_image_features=self.num_image_features,
            actor_total_state_obs_dim=self.actor_total_state_obs_dim,
            actor_core_state_obs_dim=self.actor_core_state_obs_dim,
            normalizer=normalizer,
        )
        if rnn_type == "gru":
            exporter = _ActorCriticPhasorGRUExporter(**exporter_kwargs)
        elif rnn_type in ("lstm", "lstm_sru"):
            exporter = _ActorCriticPhasorLSTMExporter(**exporter_kwargs)
        else:
            raise NotImplementedError(f"Unsupported RNN type for phasor JIT export: {rnn_type}")
        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        scripted = torch.jit.script(exporter)
        scripted.save(filepath)
        print(f"[ActorCriticPhasor] Exported JIT policy ({rnn_type}) to: {filepath}")


class _ActorCriticPhasorExporterBase(nn.Module):
    """Shared JIT-exportable actor wrapper utilities for phasor policies."""

    def __init__(
        self,
        backbone,
        attn_image_net,
        rnn,
        forward_input_layer,
        actor,
        image_input_dims,
        num_image_features,
        actor_total_state_obs_dim,
        actor_core_state_obs_dim,
        normalizer=None,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.attn_image_net = copy.deepcopy(attn_image_net)
        self.rnn = copy.deepcopy(rnn)
        self.actor = copy.deepcopy(actor)
        self.image_input_dims = tuple(image_input_dims)
        self.num_image_features = int(num_image_features)
        self.actor_total_state_obs_dim = int(actor_total_state_obs_dim)
        self.actor_core_state_obs_dim = int(actor_core_state_obs_dim)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()

        if forward_input_layer is None:
            self.actor_linear = nn.Identity()
            self.actor_activation = nn.Identity()
        else:
            self.actor_linear = copy.deepcopy(forward_input_layer.linear)
            self.actor_activation = copy.deepcopy(forward_input_layer.activation)

    def _encode_obs(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.normalizer(observations)
        actor_info = observations[..., : self.actor_total_state_obs_dim]
        state_obs = actor_info[..., : self.actor_core_state_obs_dim]
        image_obs = observations[..., self.actor_total_state_obs_dim :]
        image_obs = image_obs.reshape(-1, *self.image_input_dims)
        image_features = self.attn_image_net(image_obs, actor_info)
        _, _, _, _, _, _, sru_input = self.backbone.encode_from_state_obs(
            state_obs,
            image_features,
        )
        return sru_input


class _ActorCriticPhasorLSTMExporter(_ActorCriticPhasorExporterBase):
    """JIT-exportable phasor actor for LSTM and LSTM_SRU."""

    def __init__(self, backbone, attn_image_net, rnn, forward_input_layer, actor, image_input_dims, num_image_features, actor_total_state_obs_dim, actor_core_state_obs_dim, normalizer=None):
        super().__init__(
            backbone=backbone,
            attn_image_net=attn_image_net,
            rnn=rnn,
            forward_input_layer=forward_input_layer,
            actor=actor,
            image_input_dims=image_input_dims,
            num_image_features=num_image_features,
            actor_total_state_obs_dim=actor_total_state_obs_dim,
            actor_core_state_obs_dim=actor_core_state_obs_dim,
            normalizer=normalizer,
        )
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, observations: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()

        sru_input = self._encode_obs(observations)
        value_ring, (hidden_state, cell_state) = self.rnn(
            sru_input.unsqueeze(0),
            (self.hidden_state, self.cell_state),
        )
        self.hidden_state[:] = hidden_state
        self.cell_state[:] = cell_state

        actor_state = value_ring.squeeze(0)
        actor_input = self.actor_activation(self.actor_linear(actor_state))
        return self.actor(actor_input)

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
        self.cell_state.zero_()


class _ActorCriticPhasorGRUExporter(_ActorCriticPhasorExporterBase):
    """JIT-exportable phasor actor for GRU."""

    def __init__(self, backbone, attn_image_net, rnn, forward_input_layer, actor, image_input_dims, num_image_features, actor_total_state_obs_dim, actor_core_state_obs_dim, normalizer=None):
        super().__init__(
            backbone=backbone,
            attn_image_net=attn_image_net,
            rnn=rnn,
            forward_input_layer=forward_input_layer,
            actor=actor,
            image_input_dims=image_input_dims,
            num_image_features=num_image_features,
            actor_total_state_obs_dim=actor_total_state_obs_dim,
            actor_core_state_obs_dim=actor_core_state_obs_dim,
            normalizer=normalizer,
        )
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, observations: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()

        sru_input = self._encode_obs(observations)
        value_ring, hidden_state = self.rnn(
            sru_input.unsqueeze(0),
            self.hidden_state,
        )
        self.hidden_state[:] = hidden_state

        actor_state = value_ring.squeeze(0)
        actor_input = self.actor_activation(self.actor_linear(actor_state))
        return self.actor(actor_input)

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
