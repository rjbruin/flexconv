import torch
import numpy as np

import ckconv
from .mfn import MFNBase, gaussian_window


class MAGNet(MFNBase):
    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        steerable: bool,
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
                MAGNetLayer(
                    dim_linear,
                    hidden_channels,
                    steerable,
                    input_scale / np.sqrt(no_layers + 1),
                    alpha / (layer + 1),
                    beta,
                    bias,
                    init_spatial_value,
                )
                for layer in range(no_layers + 1)
            ]
        )


class MAGNetLayer(torch.nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        steerable: bool,
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
                (hidden_channels, dim_linear)
            )
        )
        self.input_scale = input_scale
        self.linear.weight.data *= input_scale * self.gamma.view(
            *self.gamma.shape, *((1,) * self.dim_linear)
        )
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # If steerable, create thetas
        self.steerable = steerable
        if self.steerable:
            self.theta = torch.nn.Parameter(
                torch.rand(
                    hidden_channels,
                )
            )

        return

    def forward(self, x):
        if self.steerable:
            gauss_window = rotated_gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, *((1,) * self.dim_linear)),
                self.theta,
                self.mu.view(1, *self.mu.shape, *((1,) * self.dim_linear)),
            )
        else:
            gauss_window = gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, 1, 1),
                self.mu.view(1, *self.mu.shape, 1, 1),
            )
        return gauss_window.view(1, -1, *x.shape[2:]) * torch.sin(self.linear(x))


def rotation_matrix(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(-1, 2, 2)


def rotate(theta, input):
    # theta.shape = [Out, 1]
    # input.shape = [B, Channels, 2, X, Y]
    return torch.einsum("coi, bcixy -> bcoxy", rotation_matrix(theta), input)


def rotated_gaussian_window(x, gamma, theta, mu):
    return torch.exp(
        -0.5 * ((gamma * rotate(2 * np.pi * theta, x.unsqueeze(1) - mu)) ** 2).sum(2)
    )
