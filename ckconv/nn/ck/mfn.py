import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm
import numpy as np


class MFNBase(torch.nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        weight_scale: float,
        bias: bool,
        bias_init: str,
    ):
        super().__init__()

        Linear = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        # Hidden layers
        self.linear = torch.nn.ModuleList(
            [
                Linear(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                )
                for _ in range(no_layers)
            ]
        )
        # Final layer
        self.output_linear = Linear(hidden_channels, out_channels, bias=bias)

        # Initialize
        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_channels),
                np.sqrt(weight_scale / hidden_channels),
            )

            if bias_init == "zero":
                lin.bias.data.fill_(value=0)
            elif bias_init == "uniform":
                lin.bias.data.uniform_(-1, 1)

        if bias_init == "zero":
            self.output_linear.bias.data.fill_(value=0.0)
        elif bias_init == "uniform":
            self.output_linear.bias.data.uniform_(value=0.0)

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        return out


#############################################
#       FourierNet
##############################################
class FourierLayer(torch.nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        input_scale: float,
    ):
        super().__init__()

        Linear = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        self.linear = Linear(dim_linear, hidden_channels)
        self.linear.weight.data *= input_scale
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        bias: bool,
        bias_init: str,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
    ):
        super().__init__(
            dim_linear=dim_linear,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            weight_scale=weight_scale,
            bias=bias,
            bias_init=bias_init,
        )
        self.filters = torch.nn.ModuleList(
            [
                FourierLayer(
                    dim_linear,
                    hidden_channels,
                    input_scale / np.sqrt(no_layers + 1),
                )
                for _ in range(no_layers + 1)
            ]
        )


#############################################
#       GaborNet
##############################################
class GaborNet(MFNBase):
    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        bias: bool,
        bias_init: str,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        init_spatial_value: float = 1.0,
    ):
        super().__init__(
            dim_linear=dim_linear,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            weight_scale=weight_scale,
            bias=bias,
            bias_init=bias_init,
        )
        self.filters = torch.nn.ModuleList(
            [
                GaborLayer(
                    dim_linear,
                    hidden_channels,
                    input_scale / np.sqrt(no_layers + 1),
                    alpha / (no_layers + 1),
                    beta,
                    bias,
                    init_spatial_value,
                )
                for _ in range(no_layers + 1)
            ]
        )


class GaborLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        input_scale: float,
        alpha: float,
        beta: float,
        bias: bool,
        init_spatial_value: float,
    ):
        super().__init__()

        Linear = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        self.dim_linear = dim_linear

        self.linear = Linear(dim_linear, hidden_channels, bias=bias)
        mu = init_spatial_value * (2 * torch.rand(hidden_channels, dim_linear) - 1)
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, 1)
            )  # Isotropic
        )
        self.linear.weight.data *= input_scale * self.gamma.view(
            *self.gamma.shape, *((1,) * self.dim_linear)
        )
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        gauss_window = gaussian_window(
            x,
            self.gamma.view(1, *self.gamma.shape, *((1,) * self.dim_linear)),
            self.mu.view(1, *self.mu.shape, *((1,) * self.dim_linear)),
        )
        return gauss_window * torch.sin(self.linear(x))


def gaussian_window(x, gamma, mu):
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(1) - mu)) ** 2).sum(2))
