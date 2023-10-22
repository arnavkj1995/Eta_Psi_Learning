from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hive.agents.qnets.noisy_linear import NoisyLinear
from hive.utils.utils import ActivationFn

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int],
        out_dim: int,
        hidden_units: Union[int, List[int]] = 256,
        activation_fn: ActivationFn = None,
        noisy: bool = False,
        std_init: float = 0.5,
    ):
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        
        hidden_units.append(out_dim)
        
        linear_fn = partial(NoisyLinear, std_init=std_init) if noisy else nn.Linear
        modules = [linear_fn(np.prod(in_dim), hidden_units[0])] #, activation_fn()]
        for i in range(len(hidden_units) - 1):
            modules.append(nn.ReLU())
            modules.append(linear_fn(hidden_units[i], hidden_units[i + 1]))
            
        if activation_fn is not None:
            modules.append(activation_fn())

        self.network = torch.nn.Sequential(*modules)

    def forward(self, x, flatten_dim=None):
        if flatten_dim:
            x = torch.flatten(x, start_dim=flatten_dim)
        return self.network(x)
