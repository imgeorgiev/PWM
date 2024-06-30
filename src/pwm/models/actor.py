from typing import Optional, Dict, Any, Union, Sequence, Type, List
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pwm.models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 2.0**0.5,
    ):
        super(ActorDeterministicMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        init_ = lambda m: model_utils.init(
            m,
            lambda x: nn.init.orthogonal_(x, init_gain),
            lambda x: nn.init.constant_(x, 0),
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.actor = nn.Sequential(*modules)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def forward(self, observations, deterministic=False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 1.0,
        init_logstd: float = -1.0,
        min_logstd: float = -10.0,
    ):
        super(ActorStochasticMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(nn.Identity())

        self.mu_net = nn.Sequential(*modules)

        self.logstd = torch.nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32) * init_logstd
        )

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.min_logstd = min_logstd

        for param in self.parameters():
            param.data *= init_gain

    def get_logstd(self):
        return self.logstd

    def clamp_std(self):
        self.logstd.data = torch.clamp(self.logstd.data, self.min_logstd)

    def forward(self, obs, deterministic=False):
        self.clamp_std()
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp()
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def action_log_probs(self, obs):
        self.clamp_std()
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)
        sample = dist.rsample()

        return sample, dist.log_prob(sample)

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)
