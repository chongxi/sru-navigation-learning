#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Created by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

EPSILON = 1e-7


def _align_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() > values.dim():
        mask = mask.squeeze(-1)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return mask.to(values.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    aligned_mask = _align_mask(values, mask)
    denom = aligned_mask.sum().clamp_min(1.0)
    return (values * aligned_mask).sum() / denom


def _masked_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    aligned_mask = _align_mask(values, mask)
    mean = _masked_mean(values, aligned_mask)
    var = _masked_mean((values - mean).square(), aligned_mask)
    normalized = (values - mean) / torch.sqrt(var + EPSILON)
    return torch.where(aligned_mask > 0, normalized, torch.zeros_like(values))


class SPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        **kwargs,

    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.original_learning_rate = learning_rate

        # SPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # SPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.valid_mask = infos.get(
            "transition_valid",
            torch.ones_like(dones, dtype=torch.bool, device=self.device),
        )
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, iter, max_iters):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            dropout_masks_a,
            dropout_masks_c,
            valid_mask_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            entropy_batch = self.actor_critic.entropy
                        
            if self.schedule == "adaptive":
                frac = 1.0 - iter / max_iters
                self.learning_rate = self.original_learning_rate * frac
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
                    
            # normalize advantages
            advantages_batch = _masked_normalize(advantages_batch, valid_mask_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            surrogate_term = -(
                torch.squeeze(advantages_batch) * ratio -
                torch.abs(torch.squeeze(advantages_batch)) * (ratio - 1) ** 2 / (2 * self.clip_param)
            )
            surrogate_loss = _masked_mean(surrogate_term, valid_mask_batch)
            # print("SPO max ratio: ", ratio.max().item())
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = _masked_mean(torch.max(value_losses, value_losses_clipped), valid_mask_batch) * 0.5
            else:
                value_loss = _masked_mean((returns_batch - value_batch).pow(2), valid_mask_batch) * 0.5

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * _masked_mean(entropy_batch, valid_mask_batch)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip the gradients for actor and critic separately
            nn.utils.clip_grad_norm_(self.actor_critic.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.get_critic_parameters(), self.max_grad_norm)
            
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
