import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm
import numpy as np

# Based on https://github.com/lucidrains/siren-pytorch
from torch import nn
from math import sqrt


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_linear: int,
        out_channels: int,
        hidden_channels: int,
        activation_function: str,
        norm_type: str,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        mean_init: float = 0.0,
        variance_init: float = 0.01,
        bias_value_init: float = 0.0,
        spatial_unif_init: bool = True,
    ):
        """
        Creates an 3-layer MLP, which parameterizes a convolutional kernel as:

        relative positions -> hidden_channels -> hidden_channels -> in_channels * out_channels


        :param dim_linear:  Spatial dimension of the input, e.g., for audio = 1, images = 2.
        :param out_channels:  input channels * output channels of the resulting convolutional kernel.
        :param hidden_channels: Number of hidden units.
        :param activation_function: Activation function used.
        :param norm_type: Normalization type used.
        :param bias:  If True, adds a learnable bias to the layers.
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        super().__init__()

        self.spatial_unif_init = spatial_unif_init
        self.no_layers = no_layers

        # Get Norm class
        Norm = {
            "BatchNorm": torch.nn.BatchNorm1d
            if dim_linear == 1
            else torch.nn.BatchNorm2d,
            "LayerNorm": ckconv.nn.LayerNorm,
            "": torch.nn.Identity,
        }[norm_type]

        # Get Activation Function class
        ActivationFunction = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "Swish": ckconv.nn.Swish,
        }[activation_function]

        # Define the linear layers
        Linear = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        # Construct the network
        # ---------------------
        # 1st layer:
        kernel_net = [
            Linear(dim_linear, hidden_channels, bias=bias),
            Norm(hidden_channels),
            ActivationFunction(),
        ]
        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear(
                        hidden_channels,
                        hidden_channels,
                        bias,
                    ),
                    Norm(hidden_channels),
                    ActivationFunction(),
                ]
            )
        # Last layer:
        kernel_net.extend(
            [
                Linear(hidden_channels, out_channels, bias=bias),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # initialize the kernel function
        self.initialize(
            mean=mean_init,
            variance=variance_init,
            bias_value=bias_value_init,
        )

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

    def forward(self, x):
        return self.kernel_net(x)

    def initialize(self, mean, variance, bias_value):

        # Initialize bias uniformly in [-1, 1]^(dim_layer)
        if self.spatial_unif_init:

            # Initialization of ReLUs
            net_layer = 1
            intermediate_response = None
            for (i, m) in enumerate(self.modules()):
                if (
                    isinstance(m, torch.nn.Conv1d)
                    or isinstance(m, torch.nn.Conv2d)
                    or isinstance(m, torch.nn.Linear)
                ):
                    m.weight.data.normal_(
                        mean,
                        variance,
                    )
                    if m.bias is not None:

                        if net_layer == 1:
                            # m.bias.data.fill_(bias_value)
                            range = torch.linspace(-1.0, 1.0, steps=m.weight.shape[0])
                            bias = -range * m.weight.data.clone().squeeze()
                            m.bias = torch.nn.Parameter(bias)

                            intermediate_response = [
                                m.weight.data.clone(),
                                m.bias.data.clone(),
                            ]
                            net_layer += 1

                        elif (net_layer > 1) and net_layer < (self.no_layers - 1):
                            range = torch.linspace(-1.0, 1.0, steps=m.weight.shape[0])
                            range = range + (range[1] - range[0])
                            range = (
                                range * intermediate_response[0].squeeze()
                                + intermediate_response[1]
                            )

                            bias = -torch.einsum(
                                "oi, i -> o", m.weight.data.clone().squeeze(), range
                            )
                            m.bias = torch.nn.Parameter(bias)

                            net_layer += 1

                        else:
                            m.bias.data.fill_(bias_value)

        # Else initialize all biases to bias_value
        for (i, m) in enumerate(self.modules()):
            if (
                isinstance(m, torch.nn.Conv1d)
                or isinstance(m, torch.nn.Conv2d)
                or isinstance(m, torch.nn.Linear)
            ):
                m.bias.data.fill_(bias_value)
