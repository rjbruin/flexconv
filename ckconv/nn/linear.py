import torch
import torch.nn as nn
import math


def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear3d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class MultipliedLinear1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 1 dimensional
        """
        super(MultipliedLinear1d, self).__init__(
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
        out = self.omega_0 * torch.nn.functional.conv1d(
            x, weight=self.weight, bias=None, stride=1, padding=0
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *((out.ndim - 2) * [1]))
        return out


class MultipliedLinear2d(torch.nn.Conv2d):
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
        super(MultipliedLinear2d, self).__init__(
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
        out = self.omega_0 * torch.nn.functional.conv2d(
            x, weight=self.weight, bias=None, stride=1, padding=0
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, *((out.ndim - 2) * [1]))
        return out
