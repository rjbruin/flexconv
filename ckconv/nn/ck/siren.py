import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm
import numpy as np

# Based on https://github.com/lucidrains/siren-pytorch
from torch import nn
from math import sqrt


class SIRENBase(torch.nn.Module):
    def __init__(
        self,
        dim_linear: int,
        out_channels: int,
        hidden_channels: int,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        bias_init: str,
        omega_0: float,
        learn_omega_0: bool,
        Linear_hidden: torch.nn.Module,
        Linear_out: torch.nn.Module,
    ):

        super().__init__()

        self.bias_init = bias_init

        ActivationFunction = ckconv.nn.Sine

        # Construct the network
        # ---------------------
        # 1st layer:
        kernel_net = [
            Linear_hidden(dim_linear, hidden_channels, omega_0, learn_omega_0, bias),
            ActivationFunction(),
        ]

        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear_hidden(
                        hidden_channels,
                        hidden_channels,
                        omega_0,
                        learn_omega_0,
                        bias,
                    ),
                    ActivationFunction(),
                ]
            )

        # Last layer:
        kernel_net.extend(
            [
                Linear_out(hidden_channels, out_channels, bias=bias),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # initialize the kernel function
        self.initialize(omega_0=omega_0)

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

    def forward(self, x):
        return self.kernel_net(x)

    def initialize(self, omega_0):

        net_layer = 1
        for (i, m) in enumerate(self.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    m.weight.data.uniform_(
                        -w_std, w_std
                    )  # Normally (-1, 1) / in_dim but we only use 1D inputs.
                    # Important! Bias is not defined in original SIREN implementation!
                    net_layer += 1
                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) / omega_0
                    m.weight.data.uniform_(
                        -w_std,
                        # the in_size is dim 2 in the weights of Linear and Conv layers
                        w_std,
                    )
                # Important! Bias is not defined in original SIREN implementation
                if m.bias is not None:
                    if self.bias_init == "zero":
                        m.bias.data.fill_(value=0)
                    elif self.bias_init == "uniform":
                        m.bias.data.uniform_(-1.0, 1.0)


class nSIREN(SIRENBase):
    def __init__(
        self,
        dim_linear: int,
        out_channels: int,
        hidden_channels: int,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        bias_init: str,
        omega_0: float,
        learn_omega_0: bool,
    ):
        """
        # nSIREN (new-SIREN) uses mappings of the form Sine(w0 W x + b) instead of Sine(w0[Wx + b]) as in Sitzmann et al., 2020, Romero et al., 2021.

        Creates an no-layer MLP, which parameterizes a convolutional kernel as:

        relative positions -> hidden_channels -> hidden_channels -> in_channels * out_channels


        :param dim_linear:  Spatial dimension of the input, e.g., for audio = 1, images = 2.
        :param out_channels:  input channels * output channels of the resulting convolutional kernel.
        :param hidden_channels: Number of hidden units.
        :param activation_function: Activation function used.
        :param norm_type: Normalization type used.
        :param bias:  If True, adds a learnable bias to the layers.
        :param omega_0: Value of the omega_0 value (only used in Sine networks).
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        # Get class of multiplied Linear Layers
        Linear_hidden = {
            1: ckconv.nn.MultipliedLinear1d,
            2: ckconv.nn.MultipliedLinear2d,
        }[dim_linear]

        Linear_out = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        super().__init__(
            dim_linear=dim_linear,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            weight_norm=weight_norm,
            no_layers=no_layers,
            bias=bias,
            bias_init=bias_init,
            omega_0=omega_0,
            learn_omega_0=learn_omega_0,
            Linear_hidden=Linear_hidden,
            Linear_out=Linear_out,
        )


#############################################
#       SIREN as in Sitzmann et al., 2020
##############################################


class SIREN(SIRENBase):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        dim_linear: int,
        out_channels: int,
        hidden_channels: int,
        weight_norm: bool,
        no_layers: int,
        bias: bool,
        bias_init: str,
        omega_0: float,
        learn_omega_0: bool,
    ):

        # Get class of multiplied Linear Layers
        Linear_hidden = {
            1: SIRENlayer1d,
            2: SIRENlayer2d,
        }[dim_linear]

        Linear_out = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        super().__init__(
            dim_linear=dim_linear,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            weight_norm=weight_norm,
            no_layers=no_layers,
            bias=bias,
            bias_init=bias_init,
            omega_0=omega_0,
            learn_omega_0=learn_omega_0,
            Linear_hidden=Linear_hidden,
            Linear_out=Linear_out,
        )


class SIRENlayer1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * [W x + b] as in Sitzmann et al., 2020, Romero et al., 2021,
        where x is 1 dimensional.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv1d(
            x, self.weight, self.bias, stride=1, padding=0
        )


class SIRENlayer2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 2 dimensional
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv2d(
            x, self.weight, self.bias, stride=1, padding=0
        )
