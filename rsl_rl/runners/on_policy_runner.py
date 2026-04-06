#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for PPO, SPO, and MDPO algorithms."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import MDPO, PPO, SPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticPhasor, ActorCriticRecurrent, ActorCriticSRU, EmpiricalNormalization
from rsl_rl.utils import store_code_state, VideoRecorder


class OnPolicyRunner:
    """On-policy runner for training and evaluation.

    Supports PPO, SPO, and MDPO algorithms. Automatically detects the algorithm
    type and adapts behavior accordingly (e.g., MDPO uses two actor-critics).
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Video recording (initialized lazily, enabled via set_video_recording)
        self.video_recorder: VideoRecorder | None = None

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        print("num obs", num_obs)
        print("num critic obs", num_critic_obs)

        alg_class = eval(self.alg_cfg.pop("class_name"))

        # Determine if this is MDPO (Multi-Distillation Policy Optimization)
        self.is_mdpo = alg_class == MDPO

        if self.is_mdpo:
            # MDPO uses two actor-critics
            actor_critic_1: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            actor_critic_2: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            self.alg = alg_class(actor_critic_1, actor_critic_2, device=self.device, **self.alg_cfg)
        else:
            # Standard algorithms use one actor-critic
            actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticSRU = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self._staggered_env_bucket_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._staggered_bucket_phases = torch.empty(0, dtype=torch.long, device=self.device)
        self._staggered_max_rollout_phases = 0
        self._maybe_compile_policy()

    def _maybe_compile_actor_critic(self, actor_critic):
        """Compile supported actor-critic hot paths when requested."""
        if not self.cfg.get("torch_compile_policy", False):
            return
        if not hasattr(torch, "compile"):
            print("[WARN] torch.compile is unavailable in this PyTorch build; skipping policy compilation.")
            return
        mode = self.cfg.get("torch_compile_mode", "default")
        compile_kwargs = {}
        if mode and mode != "default":
            compile_kwargs["mode"] = mode
        try:
            if isinstance(actor_critic, ActorCriticSRU):
                actor_critic._compute_action_mean = torch.compile(actor_critic._compute_action_mean, **compile_kwargs)
                actor_critic._compute_value = torch.compile(actor_critic._compute_value, **compile_kwargs)
            elif hasattr(actor_critic, "update_distribution"):
                actor_critic.update_distribution = torch.compile(actor_critic.update_distribution, **compile_kwargs)
                if hasattr(actor_critic, "evaluate"):
                    actor_critic.evaluate = torch.compile(actor_critic.evaluate, **compile_kwargs)
            print(
                f"[INFO] torch.compile enabled for {actor_critic.__class__.__name__}"
                f" (mode={mode})"
            )
        except Exception as exc:
            print(
                f"[WARN] torch.compile failed for {actor_critic.__class__.__name__}: {exc}."
                " Continuing without compilation."
            )

    def _maybe_compile_policy(self):
        """Compile supported policy hot paths for the current algorithm."""
        if self.is_mdpo:
            self._maybe_compile_actor_critic(self.alg.actor_critic_1)
            self._maybe_compile_actor_critic(self.alg.actor_critic_2)
        else:
            self._maybe_compile_actor_critic(self.alg.actor_critic)

    def _reset_policy_memories(self, dones: torch.Tensor):
        """Reset recurrent policy memory for envs that ended during warm-up."""
        if self.is_mdpo:
            self.alg.actor_critic_1.reset(dones[self.alg.indices_1])
            self.alg.actor_critic_2.reset(dones[self.alg.indices_2])
        else:
            self.alg.actor_critic.reset(dones)

    def _clear_warmup_transitions(self):
        """Drop any temporary transition data generated during stagger warm-up."""
        if self.is_mdpo:
            self.alg.transition_1.clear()
            self.alg.transition_2.clear()
            self.alg.storage_1.clear()
            self.alg.storage_2.clear()
        else:
            self.alg.transition.clear()
            self.alg.storage.clear()

    def _reset_staggered_gate_schedule(self):
        """Clear the runner-side staggered gate schedule."""
        self._staggered_env_bucket_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._staggered_bucket_phases = torch.empty(0, dtype=torch.long, device=self.device)
        self._staggered_max_rollout_phases = 0

    def _bucket_env_ids(self, bucket_idx: int) -> torch.Tensor:
        """Return env ids currently assigned to a staggered reset bucket."""
        if self._staggered_env_bucket_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return (self._staggered_env_bucket_ids == bucket_idx).nonzero(as_tuple=False).squeeze(-1)

    def _apply_staggered_reset_gate(self) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Apply the next reset gate and return fresh normalized observations and gate logs."""
        if self._staggered_bucket_phases.numel() == 0:
            obs, extras = self.env.get_observations()
            return (*self._compute_normalized_observations(obs, extras), None)

        self._staggered_bucket_phases += 1
        due_bucket_ids = (self._staggered_bucket_phases >= self._staggered_max_rollout_phases).nonzero(as_tuple=False).squeeze(-1)
        gate_log = None
        for bucket_idx in due_bucket_ids.tolist():
            gate_env_ids = self._bucket_env_ids(bucket_idx)
            reset_env_ids = self.env.unwrapped.apply_staggered_reset_gate(gate_env_ids)
            if reset_env_ids.numel() > 0:
                self._staggered_env_bucket_ids[reset_env_ids] = bucket_idx
                manual_dones = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
                manual_dones[reset_env_ids] = 1
                self._reset_policy_memories(manual_dones)
                if "log" in self.env.unwrapped.extras and self.env.unwrapped.extras["log"]:
                    gate_log = dict(self.env.unwrapped.extras["log"])
                    self.env.unwrapped.extras["log"] = dict()
            self._staggered_bucket_phases[bucket_idx] = 0

        obs, extras = self.env.get_observations()
        return (*self._compute_normalized_observations(obs, extras), gate_log)

    def _compute_normalized_observations(
        self, obs: torch.Tensor, extras: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize current actor and critic observations on the runner device."""
        critic_obs = extras["observations"].get("critic", obs)
        obs = self.obs_normalizer(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return obs.to(self.device), critic_obs.to(self.device)

    def _apply_staggered_resets(self, requested_buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance envs in rollout-length chunks so batches start with mixed task phases."""
        max_episode_length = int(self.env.max_episode_length)
        max_buckets = max(1, (max_episode_length + self.num_steps_per_env - 1) // self.num_steps_per_env)
        num_buckets = min(max(1, requested_buckets), self.env.num_envs, max_buckets)

        if num_buckets <= 1:
            self._reset_staggered_gate_schedule()
            self.env.unwrapped.configure_staggered_reset_gating(False)
            return self._compute_normalized_observations(*self.env.reset())

        print(
            "[INFO] Applying staggered resets:"
            f" buckets={num_buckets}, rollout_len={self.num_steps_per_env}, max_episode_length={max_episode_length}"
        )

        max_rollout_phases = max_buckets
        target_bucket_phases = torch.div(
            torch.arange(num_buckets, device=self.device, dtype=torch.long) * max_rollout_phases,
            num_buckets,
            rounding_mode="floor",
        )
        warmup_chunks = int(target_bucket_phases.max().item())
        warmup_env_steps = warmup_chunks * self.num_steps_per_env
        print(
            "[INFO] Staggered warm-up:"
            f" chunks={warmup_chunks}, env_steps={warmup_env_steps}, bucket_phases={target_bucket_phases.tolist()}"
        )
        env_bucket_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long) % num_buckets
        self._staggered_env_bucket_ids = env_bucket_ids
        self._staggered_bucket_phases = target_bucket_phases.clone()
        self._staggered_max_rollout_phases = max_rollout_phases

        self.env.unwrapped.configure_staggered_reset_gating(True)
        obs, extras = self.env.reset()
        obs, critic_obs = self._compute_normalized_observations(obs, extras)

        with torch.no_grad():
            max_target_phase = int(target_bucket_phases.max().item())
            for chunk_idx in tqdm(range(max_target_phase), desc="Stagger warmup", leave=False):
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, _, dones, infos = self.env.step(actions)
                    dones = dones.to(self.device)
                    self._reset_policy_memories(dones)
                    obs, critic_obs = self._compute_normalized_observations(obs, infos)

                due_phase = max_target_phase - (chunk_idx + 1)
                due_bucket_ids = (target_bucket_phases == due_phase).nonzero(as_tuple=False).squeeze(-1)
                for bucket_idx in due_bucket_ids.tolist():
                    reset_env_ids = self.env.unwrapped.apply_staggered_reset_gate(self._bucket_env_ids(bucket_idx))
                    if reset_env_ids.numel() > 0:
                        self._staggered_env_bucket_ids[reset_env_ids] = bucket_idx
                        manual_dones = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
                        manual_dones[reset_env_ids] = 1
                        self._reset_policy_memories(manual_dones)
                obs, extras = self.env.get_observations()
                obs, critic_obs = self._compute_normalized_observations(obs, extras)

        self._clear_warmup_transitions()
        self.alg.reset_dropout_masks()
        return obs, critic_obs

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        staggered_reset_buckets: int = 0,
    ):
        """Run the training loop.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: If True, randomize initial episode lengths.
            staggered_reset_buckets: Number of rollout-phase buckets to create by pre-rolling envs.
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        self.train_mode()

        if staggered_reset_buckets > 1:
            obs, critic_obs = self._apply_staggered_resets(staggered_reset_buckets)
        else:
            self._reset_staggered_gate_schedule()
            if hasattr(self.env.unwrapped, "configure_staggered_reset_gating"):
                self.env.unwrapped.configure_staggered_reset_gating(False)
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            obs, extras = self.env.get_observations()
            critic_obs = extras["observations"].get("critic", obs)
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Get reward shifting value for MDPO
        reward_shifting_value = self.cfg.get("reward_shifting_value", 0.0)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Check if we should start recording video this iteration
            if self.video_recorder:
                # Debug: print every 100 iterations to track progress
                if it % 100 == 0:
                    print(f"[DEBUG VideoRecorder] Iteration {it}, checking should_record...")
                if self.video_recorder.should_record(it):
                    self.video_recorder.start_recording()

            # Rollout
            with torch.no_grad():
                for i in tqdm(range(self.num_steps_per_env), desc=f"Iter {it}/{tot_iter} rollout", leave=False):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)

                    # Capture video frame if recording
                    # Continue capturing until video_length is reached
                    if self.video_recorder and self.video_recorder.is_recording:
                        self.video_recorder.capture_frame()

                    # Apply reward shifting for MDPO
                    if self.is_mdpo and reward_shifting_value != 0.0:
                        rewards = rewards + reward_shifting_value

                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    transition_valid = ~infos.get(
                        "staggered_invalid",
                        torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
                    )
                    infos["transition_valid"] = transition_valid
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos and infos["episode"]:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos and infos["log"]:
                            ep_infos.append(infos["log"])

                        # Shift rewards back for logging
                        log_rewards = rewards
                        if self.is_mdpo and reward_shifting_value != 0.0:
                            log_rewards = rewards - reward_shifting_value

                        cur_reward_sum[transition_valid] += log_rewards[transition_valid]
                        cur_episode_length[transition_valid] += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                if self._staggered_bucket_phases.numel() > 0:
                    obs, critic_obs, gate_log = self._apply_staggered_reset_gate()
                    if self.log_dir is not None and gate_log:
                        ep_infos.append(gate_log)
                    stop = time.time()
                    collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

                # update dropout masks
                self.alg.update_dropout_masks()

            # Update returns different values based on algorithm type
            update_result = self.alg.update(it, tot_iter)
            if self.is_mdpo:
                if len(update_result) == 4:
                    mean_value_loss, mean_surrogate_loss, mean_kl_divergence, mean_entropy = update_result
                else:
                    mean_value_loss, mean_surrogate_loss, mean_kl_divergence = update_result
                    mean_entropy = None
            else:
                if len(update_result) == 3:
                    mean_value_loss, mean_surrogate_loss, mean_entropy = update_result
                else:
                    mean_value_loss, mean_surrogate_loss = update_result
                    mean_entropy = None
                mean_kl_divergence = None

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # reset dropout masks
            self.alg.reset_dropout_masks()

            if self.log_dir is not None:
                self.log(locals())

                # Log video only if recording is complete (reached video_length frames)
                if self.video_recorder and self.video_recorder.is_recording and self.video_recorder.is_complete():
                    self.video_recorder.log_video(self.writer, it, self.logger_type)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training statistics."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            ep_info_keys = set()
            for ep_info in locs["ep_infos"]:
                ep_info_keys.update(ep_info.keys())
            for key in sorted(ep_info_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # Get action std from appropriate actor-critic
        if self.is_mdpo:
            action_std = self.alg.actor_critic_1.action_std
        else:
            action_std = self.alg.actor_critic.action_std
        if action_std.ndim == 1:
            per_action_std = action_std
        else:
            per_action_std = action_std.reshape(-1, action_std.shape[-1]).mean(dim=0)
        mean_std = per_action_std.mean()

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        if self.is_mdpo and locs["mean_kl_divergence"] is not None:
            self.writer.add_scalar("Loss/kl_divergence", locs["mean_kl_divergence"], locs["it"])
        if locs.get("mean_entropy") is not None:
            self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        for action_idx, std in enumerate(per_action_std):
            self.writer.add_scalar(f"Policy/noise_std_action_{action_idx}", std.item(), locs["it"])
        if per_action_std.numel() >= 2:
            self.writer.add_scalar("Policy/noise_std_vx", per_action_std[0].item(), locs["it"])
            self.writer.add_scalar("Policy/noise_std_steering", per_action_std[1].item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        log_str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{log_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )
            if self.is_mdpo and locs["mean_kl_divergence"] is not None:
                log_string += f"""{'KL divergence:':>{pad}} {locs['mean_kl_divergence']:.4f}\n"""
            if locs.get("mean_entropy") is not None:
                log_string += f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            if per_action_std.numel() >= 2:
                log_string += f"""{'Action noise std [vx, steering]:':>{pad}} [{per_action_std[0].item():.2f}, {per_action_std[1].item():.2f}]\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{log_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            if per_action_std.numel() >= 2:
                log_string += f"""{'Action noise std [vx, steering]:':>{pad}} [{per_action_std[0].item():.2f}, {per_action_std[1].item():.2f}]\n"""
            if locs.get("mean_entropy") is not None:
                log_string += f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        """Save the model checkpoint."""
        if self.is_mdpo:
            saved_dict = {
                "model_state_dict": self.alg.actor_critic_1.state_dict(),
                "model_state_dict_1": self.alg.actor_critic_1.state_dict(),
                "model_state_dict_2": self.alg.actor_critic_2.state_dict(),
                "optimizer_state_dict": self.alg.optimizer_1.state_dict(),
                "optimizer_state_dict_1": self.alg.optimizer_1.state_dict(),
                "optimizer_state_dict_2": self.alg.optimizer_2.state_dict(),
                "algorithm_class_name": self.alg.__class__.__name__,
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        else:
            saved_dict = {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "algorithm_class_name": self.alg.__class__.__name__,
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        """Load a model checkpoint."""
        loaded_dict = torch.load(path, weights_only=True)
        if self.is_mdpo:
            model_state_dict_1 = loaded_dict.get("model_state_dict_1", loaded_dict["model_state_dict"])
            model_state_dict_2 = loaded_dict.get("model_state_dict_2", loaded_dict.get("model_state_dict_1", loaded_dict["model_state_dict"]))
            self.alg.actor_critic_1.load_state_dict(model_state_dict_1, strict=True)
            self.alg.actor_critic_2.load_state_dict(model_state_dict_2, strict=True)
            if load_optimizer:
                optimizer_state_dict_1 = loaded_dict.get("optimizer_state_dict_1", loaded_dict["optimizer_state_dict"])
                optimizer_state_dict_2 = loaded_dict.get("optimizer_state_dict_2", loaded_dict.get("optimizer_state_dict_1", loaded_dict["optimizer_state_dict"]))
                self.alg.optimizer_1.load_state_dict(optimizer_state_dict_1)
                self.alg.optimizer_2.load_state_dict(optimizer_state_dict_2)
        else:
            model_state_dict = loaded_dict.get("model_state_dict_1", loaded_dict["model_state_dict"])
            self.alg.actor_critic.load_state_dict(model_state_dict, strict=True)
            if load_optimizer:
                optimizer_state_dict = loaded_dict.get("optimizer_state_dict_1", loaded_dict["optimizer_state_dict"])
                self.alg.optimizer.load_state_dict(optimizer_state_dict)
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Get the inference policy function."""
        self.eval_mode()
        if self.is_mdpo:
            actor_critic = self.alg.actor_critic_1
        else:
            actor_critic = self.alg.actor_critic

        if device is not None:
            actor_critic.to(device)
        policy = actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def get_policy_reset(self, device=None):
        """Get the policy reset function."""
        self.eval_mode()
        if self.is_mdpo:
            actor_critic = self.alg.actor_critic_1
        else:
            actor_critic = self.alg.actor_critic

        if device is not None:
            actor_critic.to(device)
        return actor_critic.reset

    def train_mode(self):
        """Switch to training mode."""
        if self.is_mdpo:
            self.alg.train_mode()
        else:
            self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        if self.is_mdpo:
            self.alg.test_mode()
        else:
            self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """Add a git repository to track for logging."""
        self.git_status_repos.append(repo_file_path)

    def set_video_recording(
        self, enable: bool, video_length: int = 200, video_interval: int = 2000, fps: int = 30, save_local: bool = True
    ):
        """Configure video recording during training.

        Video recording can upload to WandB and/or save locally as MP4 files.

        Requirements:
        1. Environment with render_mode="rgb_array"
        2. For WandB upload: logger="wandb" in agent config
        3. For local save: imageio package installed

        Args:
            enable: Whether to enable video recording.
            video_length: Number of environment steps per video. Defaults to 200.
            video_interval: Number of training iterations between video recordings. Defaults to 2000.
            fps: Frames per second for the recorded video. Defaults to 30.
            save_local: Whether to save videos locally as MP4 files. Defaults to True.
        """
        if enable:
            self.video_recorder = VideoRecorder(
                env=self.env,
                video_length=video_length,
                video_interval=video_interval,
                fps=fps,
                save_local=save_local,
                log_dir=self.log_dir,
            )
            self.video_recorder.enable()
        else:
            if self.video_recorder:
                self.video_recorder.disable()
            self.video_recorder = None
