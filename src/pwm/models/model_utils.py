from torch import vmap
import torch.nn as nn
from functorch import combine_state_for_ensemble


def init(module, weight_init, bias_init):
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr
