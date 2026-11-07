"""rsl_rl pieces: the CNN+GRU actor, a memory-safe recurrent minibatch generator, a no-op log writer.

``CNNGRUModel``: depth stack -> CNN -> concat proprio -> GRU -> MLP -> Gaussian. rsl-rl-lib 5.4.1 has
CNNModel and RNNModel but no combination; this subclasses RNNModel so PPO's recurrent plumbing (hidden-state
storage, reset on done, BPTT minibatches) is inherited unchanged. Selected by class path in the train cfg.
"""
from __future__ import annotations

from typing import Any, Generator

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import CNN, RNN, HiddenState
from tensordict import TensorDict


class CNNGRUModel(RNNModel):
    """CNN encoder(s) for the 2-D observation groups + GRU core + MLP head."""

    def __init__(self, obs: TensorDict, obs_groups: dict, obs_set: str, output_dim: int,
                 hidden_dims=(256, 256, 256), activation: str = 'elu', obs_normalization: bool = False,
                 distribution_cfg: dict | None = None, cnn_cfg: dict[str, Any] | None = None, cnns=None,
                 rnn_type: str = 'gru', rnn_hidden_dim: int = 256, rnn_num_layers: int = 1) -> None:
        # Splits groups into 1-D / 2-D exactly as CNNModel does (sets obs_groups_2d, obs_dims_2d, ...).
        CNNModel._get_obs_dim(self, obs, obs_groups, obs_set)
        if cnns is None:
            if cnn_cfg is None:
                raise ValueError('cnn_cfg is required')
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {g: cnn_cfg for g in self.obs_groups_2d}
            cnns = {g: CNN(input_dim=self.obs_dims_2d[i], input_channels=self.obs_channels_2d[i], **cnn_cfg[g])
                    for i, g in enumerate(self.obs_groups_2d)}
        self.cnn_latent_dim = sum(int(c.output_dim) for c in cnns.values())
        self.latent_dim = rnn_hidden_dim
        # RNNModel.__init__ is skipped on purpose: it sizes the GRU from the 1-D obs width only.
        MLPModel.__init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                          distribution_cfg)
        self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.cnns = cnns if isinstance(cnns, nn.ModuleDict) else nn.ModuleDict(cnns)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                   hidden_state: HiddenState = None) -> torch.Tensor:
        latents = []
        for g in self.obs_groups_2d:                 # rollout [N,C,H,W]; update [T,N,C,H,W] -> fold for Conv2d
            img = obs[g]
            lead = img.shape[:-3]
            z = self.cnns[g](img.reshape(-1, *img.shape[-3:]) if len(lead) > 1 else img)
            latents.append(z.reshape(*lead, z.shape[-1]) if len(lead) > 1 else z)
        latent = torch.cat(latents, dim=-1)
        if self.obs_groups:
            latent = torch.cat([MLPModel.get_latent(self, obs), latent], dim=-1)
        return self.rnn(latent, masks, hidden_state).squeeze(0)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        return CNNModel._get_obs_dim(self, obs, obs_groups, obs_set)

    def as_jit(self):
        raise NotImplementedError('RNNModel exporters would drop the CNN encoder')

    as_onnx = as_jit


def memory_safe_recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
    """Drop-in for RolloutStorage.recurrent_mini_batch_generator that pads one minibatch at a time.

    The stock version pads every trajectory of the whole rollout at once (T x n_trajectories x obs); with
    short early episodes that is tens of GiB. Minibatches are groups of envs, so padding per minibatch yields
    identical batches with num_mini_batches x less peak memory."""
    from rsl_rl.storage.rollout_storage import RolloutStorage
    from rsl_rl.utils import split_and_pad_trajectories
    mb = self.num_envs // num_mini_batches
    dones = self.dones.squeeze(-1)
    last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    last_was_done[1:] = dones[:-1]
    last_was_done[0] = True
    lwd = last_was_done.permute(1, 0)                          # (envs, time)

    def initial_hidden(saved, start, stop):
        if saved is None:
            return None
        out = [s.permute(2, 0, 1, 3)[start:stop][lwd[start:stop]].transpose(1, 0).contiguous() for s in saved]
        return out[0] if len(out) == 1 else out

    for _ in range(num_epochs):
        for i in range(num_mini_batches):
            start, stop = i * mb, (i + 1) * mb
            padded_obs, masks = split_and_pad_trajectories(self.observations[:, start:stop], self.dones[:, start:stop])
            yield RolloutStorage.Batch(
                observations=padded_obs, actions=self.actions[:, start:stop], values=self.values[:, start:stop],
                advantages=self.advantages[:, start:stop], returns=self.returns[:, start:stop],
                old_actions_log_prob=self.actions_log_prob[:, start:stop],
                old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),
                hidden_states=(initial_hidden(self.saved_hidden_state_a, start, stop),
                               initial_hidden(self.saved_hidden_state_c, start, stop)),
                masks=masks)


def patch_rsl_rl() -> None:
    from rsl_rl.storage.rollout_storage import RolloutStorage
    RolloutStorage.recurrent_mini_batch_generator = memory_safe_recurrent_mini_batch_generator


class NullLogWriter:
    """rsl_rl has no 'logging off'; this writer accepts every call and does nothing (metrics come from the env)."""

    def __init__(self, log_dir=None, **kwargs) -> None:
        self.log_dir = log_dir

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None
