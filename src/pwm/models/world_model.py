# adapted from TDMPC2 repository but modified to support
# differentiating through reward two hot inversion
# https://github.com/nicklashansen/tdmpc2/

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import SimNorm, mlp


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


DREG_BINS = None


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(
        self,
        observation_dim,
        action_dim,
        latent_dim,
        units,
        encoder_units,
        encoder,
        dynamics,
        reward,
        action_dims=None,
        num_bins=None,
        vmin=None,
        vmax=None,
        multitask=False,
        tasks=None,
        task_dim=0,
    ):
        super().__init__()
        self.multitask = multitask
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        if self.multitask:
            self._task_emb = nn.Embedding(len(tasks), task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(tasks), action_dim)
            for i in range(len(tasks)):
                self._action_masks[i, : action_dims[i]] = 1.0
        self._encoder = mlp(
            observation_dim + task_dim,
            encoder_units,
            latent_dim,
            last_layer=encoder["last_layer"],
            last_layer_kwargs=encoder["last_layer_kwargs"],
        )
        self._dynamics = mlp(
            latent_dim + action_dim + task_dim,
            units,
            latent_dim,
            last_layer=dynamics["last_layer"],
            last_layer_kwargs=dynamics["last_layer_kwargs"],
        )
        self._reward = mlp(
            latent_dim + action_dim + task_dim,
            units,
            max(num_bins, 1) if num_bins else 1,
            last_layer=reward["last_layer"],
            last_layer_kwargs=reward["last_layer_kwargs"],
        )
        self.apply(weight_init)
        zero_([self._reward[-1].weight])

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        return self

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.multitask:
            obs = self.task_emb(obs, task)
        return self._encoder(obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def step(self, z, a, task):
        """
        Predicts the next latent state, reward, and termination signal.
        """
        assert z.shape[0] == a.shape[0]
        z_next = self.next(z, a, task)
        r = self.reward(z, a, task)
        return z_next, r

    def two_hot_inv(self, x):
        """Converts a batch of soft two-hot encoded vectors to scalars."""
        global DREG_BINS
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        if DREG_BINS is None:
            DREG_BINS = torch.linspace(
                self.vmin, self.vmax, self.num_bins, device=x.device
            )
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
        return symexp(x)

    def almost_two_hot_inv(self, x):
        """Converts a batch of soft two-hot encoded vectors to scalars."""
        if self.num_bins == 0 or self.num_bins == None:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        # TODO this computation below can probably be optimized
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return x
