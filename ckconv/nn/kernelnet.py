import torch
import ckconv
from torch.nn.utils import weight_norm as w_norm
import numpy as np

# Based on https://github.com/lucidrains/siren-pytorch
from torch import nn
from math import sqrt
from torch.profiler import record_function


class KernelNet(torch.nn.Module):
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
        omega_0: float,
        learn_omega_0: bool,
        weight_dropout: float,
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
        :param omega_0: Value of the omega_0 value (only used in Sine networks).
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        super().__init__()

        is_siren = activation_function == "Sine"
        w_dp = weight_dropout != 0.0

        Norm = {
            "BatchNorm": torch.nn.BatchNorm1d,
            "LayerNorm": ckconv.nn.LayerNorm,
            "": torch.nn.Identity,
        }[norm_type]
        # If the network is a SIREN, no Norm will be used.
        if is_siren:
            Norm = torch.nn.Identity

        ActivationFunction = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "Swish": ckconv.nn.Swish,
            "Sine": ckconv.nn.Sine,
        }[activation_function]

        Linear_hidden = {
            1: ckconv.nn.MultipliedLinear1d,
            2: ckconv.nn.MultipliedLinear2d,
        }[dim_linear]

        Linear_out = {
            1: ckconv.nn.Linear1d,
            2: ckconv.nn.Linear2d,
        }[dim_linear]

        # Construct the network
        # ---------------------
        # 1st layer:
        kernel_net = [
            Linear_hidden(
                dim_linear, hidden_channels, omega_0, learn_omega_0, bias=bias
            ),
            Norm(hidden_channels),
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
                    Norm(hidden_channels),
                    ActivationFunction(),
                ]
            )

        # Last layer:
        kernel_net.extend(
            [
                Linear_out(hidden_channels, out_channels, bias=bias),
                torch.nn.Dropout(p=weight_dropout) if w_dp else torch.nn.Identity(),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # initialize the kernel function
        self.initialize(
            mean=0.0,
            variance=0.01,  # TODO: Only used if not is_siren
            bias_value=0.0,
            is_siren=is_siren,
            omega_0=omega_0,
        )

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

    def forward(self, x):
        return self.kernel_net(x)

    def initialize(self, mean, variance, bias_value, is_siren, omega_0):

        if is_siren:
            # Initialization of SIRENs
            net_layer = 1
            for (i, m) in enumerate(self.modules()):
                if (
                    isinstance(m, torch.nn.Conv1d)
                    or isinstance(m, torch.nn.Conv2d)
                    or isinstance(m, torch.nn.Linear)
                    or isinstance(m, ckconv.MultipliedLinear2d)
                ):
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
                        m.bias.data.uniform_(-1.0, 1.0)
        else:
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

                        elif net_layer == 2:
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


class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(
        self,
        w0,
        learn_omega_0,
    ):
        super().__init__()
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(w0)
        else:
            self.omega_0 = w0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.
    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0,
        use_bias,
        learn_omega_0,
        activation=None,
        c=6.0,
        is_first=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = ckconv.nn.Linear2d(dim_in, dim_out, bias=use_bias)
        # self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0, learn_omega_0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
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
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0,
        w0_initial,
        use_bias,
        final_activation,
        learn_omega_0,
        weight_norm,
    ):
        super().__init__()
        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    learn_omega_0=learn_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

        final_activation = (
            nn.Identity() if final_activation is None else final_activation
        )
        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            learn_omega_0=learn_omega_0,
        )

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.net):
                if isinstance(module, (SirenLayer)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.net[i].linear = w_norm(module.linear)

            self.last_layer.linear = w_norm(self.last_layer.linear)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        weight_scale: float,
        bias: bool,
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [
                ckconv.nn.Linear2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                )
                for _ in range(no_layers)
            ]
        )
        self.output_linear = ckconv.nn.Linear2d(
            hidden_channels, out_channels, bias=bias
        )

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_channels),
                np.sqrt(weight_scale / hidden_channels),
            )

        return

    def forward(self, x):
        with record_function("mfn_base"):
            with record_function("filter_0"):
                out = self.filters[0](x)
            for i in range(1, len(self.filters)):
                with record_function(f"linear_{i}"):
                    lin = self.linear[i - 1](out)
                with record_function(f"filter_{i}"):
                    out = self.filters[i](x) * lin
            with record_function(f"linear_out"):
                out = self.output_linear(out)

            return out


class GaborNet(MFNBase):
    def __init__(
        self,
        dim_linear: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        steerable: bool,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        bias: bool = True,
        init_spatial_value: float = 1.0,
    ):
        super().__init__(
            hidden_channels,
            out_channels,
            no_layers,
            weight_scale,
            bias,
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
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


class GaborLayer(nn.Module):
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
        self.linear = ckconv.nn.Linear2d(dim_linear, hidden_channels, bias=bias)
        mu = init_spatial_value * (2 * torch.rand(hidden_channels, dim_linear) - 1)
        self.mu = nn.Parameter(mu)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample(
                (hidden_channels, dim_linear)
            )
        )
        self.input_scale = input_scale
        self.linear.weight.data *= input_scale * self.gamma.view(
            *self.gamma.shape, 1, 1
        )
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # If steerable, create thetas
        self.steerable = steerable
        if self.steerable:
            self.theta = nn.Parameter(
                torch.rand(
                    hidden_channels,
                )
            )

        return

    def forward(self, x):
        with record_function(f"gaussian_window"):
            if self.steerable:
                gauss_window = rotated_gaussian_window(
                    x,
                    self.gamma.view(1, *self.gamma.shape, 1, 1),
                    self.theta,
                    self.mu.view(1, *self.mu.shape, 1, 1),
                )
            else:
                gauss_window = gaussian_window(
                    x,
                    self.gamma.view(1, *self.gamma.shape, 1, 1),
                    self.mu.view(1, *self.mu.shape, 1, 1),
                )
        with record_function(f"gabor_filter"):
            return gauss_window * torch.sin(self.linear(x))


def gaussian_window(x, gamma, mu):
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(1) - mu)) ** 2).sum(2))


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
