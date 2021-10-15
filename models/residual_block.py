# torch
import torch

# typing
from functools import partial
from typing import Tuple, Union
from ckconv.nn import FlexConv, CKConv
from torch.nn import Conv1d, Conv2d


class ResidualBlockBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ConvType: Union[CKConv, FlexConv, Conv1d, Conv2d],
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
        dropout: float,
    ):
        """
        Instantiates the core elements of a residual block but does not implement the forward function.
        These elements are:
        (1) Two convolutional layers
        (2) Two normalization layers
        (3) A residual connection
        (4) A dropout layer
        """
        super().__init__()

        # Conv Layers
        self.cconv1 = ConvType(in_channels=in_channels, out_channels=out_channels)
        self.cconv2 = ConvType(in_channels=out_channels, out_channels=out_channels)

        # Nonlinear layer
        self.nonlinear = NonlinearType()

        # Norm layers
        self.norm1 = NormType(out_channels)
        self.norm2 = NormType(out_channels)

        # Dropout
        self.dp = torch.nn.Dropout(dropout)

        # Shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(LinearType(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        raise NotImplementedError()
