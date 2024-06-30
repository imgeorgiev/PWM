from typing import Type, List
import torch
import torch.nn as nn

from pwm.models import model_utils


class CriticMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 2.0**0.5,
    ):
        super(CriticMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [1]

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
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules)

        self.obs_dim = obs_dim

    def forward(self, observations):
        return self.critic(observations)

    def predict(self, observations):
        return self.forward(observations)


class DoubleCriticMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 2.0**0.5,
    ):
        super(DoubleCriticMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [1]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        init_ = lambda m: model_utils.init(
            m,
            lambda x: nn.init.orthogonal_(x, gain=init_gain),
            lambda x: nn.init.constant_(x, 0),
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic_1 = nn.Sequential(*modules)
        self.critic_2 = nn.Sequential(*modules)

        self.obs_dim = obs_dim

    def forward(self, observations):
        v1 = self.critic_1(observations)
        v2 = self.critic_2(observations)
        return torch.min(v1, v2)

    def predict(self, observations):
        """Different from forward as it returns both critic values estimates"""
        v1 = self.critic_1(observations)
        v2 = self.critic_2(observations)
        return torch.cat((v1, v2), dim=-1)
