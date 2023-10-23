import torch
from torch import nn

import numpy as np

from hive.agents.qnets.mlp import MLPNetwork
from hive.agents.qnets.utils import calculate_output_dim

class ConvNet(nn.Module):
    def __init__(
        self,
        in_dim,
        channels=None,
        mlp_layers=None,
        kernel_sizes=1,
        strides=1,
        paddings=0,
        normalization_factor=255
    ):
        super().__init__()
        self._normalization_factor = normalization_factor
        if channels is not None:
            if isinstance(kernel_sizes, int):
                kernel_sizes = [kernel_sizes] * len(channels)
            if isinstance(strides, int):
                strides = [strides] * len(channels)
            if isinstance(paddings, int):
                paddings = [paddings] * len(channels)

            if not all(
                len(x) == len(channels) for x in [kernel_sizes, strides, paddings]
            ):
                raise ValueError("The lengths of the parameter lists must be the same")

            # Convolutional Layers
            channels = [in_dim[0], *channels]
            conv_seq = []
            for i in range(0, len(channels) - 1):
                conv_seq.append(
                    torch.nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                    )
                )
                conv_seq.append(torch.nn.ReLU())
            self.conv = torch.nn.Sequential(*conv_seq)
        else:
            self.conv = torch.nn.Identity()

        mlp_seq = []
        conv_output_size = calculate_output_dim(self.conv, in_dim)
        mlp_layers = [int(np.prod(conv_output_size)), *mlp_layers]
        
        for i in range(0, len(mlp_layers) - 1):
            mlp_seq.append(
                torch.nn.Linear(mlp_layers[i], mlp_layers[i + 1]
                )
            )
            if i != len(mlp_layers) - 2:
                mlp_seq.append(torch.nn.ReLU())
    
        self.mlp = torch.nn.Sequential(*mlp_seq)

    def forward(self, x):
        sh = x.shape
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = x.float()
        x = x / self._normalization_factor
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x) #, flatten_dim=1)
        return x #.reshape(sh[:-3] + x.shape[1:])
