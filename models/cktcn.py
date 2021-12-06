import torch
import ckconv.nn
from functools import partial
from .residual_block import ResidualBlockBase

# typing
from omegaconf import OmegaConf
from typing import Tuple, Union
from ckconv.nn import FlexConv, CKConv
from torch.nn import Conv1d, Conv2d


class TCNBlock(ResidualBlockBase):
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
        Creates a Residual Block as in TCNs ( Bai et. al., 2017 )

        input
         | ---------------|
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         |                |
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         + <--------------|
         |
         ReLU
         |
         output
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ConvType=ConvType,
            NonlinearType=NonlinearType,
            NormType=NormType,
            LinearType=LinearType,
            dropout=dropout,
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(self.nonlinear(self.norm1(self.cconv1(x))))
        out = self.nonlinear(self.dp(self.nonlinear(self.norm2(self.cconv2(out)))) + shortcut)
        return out


class TCNBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        mask_config: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_config.no_hidden
        no_blocks = net_config.no_blocks
        norm = net_config.norm
        dropout = net_config.dropout
        block_width_factors = net_config.block_width_factors
        nonlinearity = net_config.nonlinearity

        # Unpack kernel_config
        kernel_type = kernel_config.type
        kernel_dim_linear = kernel_config.dim_linear
        kernel_no_hidden = kernel_config.no_hidden
        kernel_no_layers = kernel_config.no_layers
        kernel_activ_function = kernel_config.activ_function
        kernel_norm = kernel_config.norm
        kernel_omega_0 = kernel_config.omega_0
        kernel_learn_omega_0 = kernel_config.learn_omega_0
        kernel_weight_norm = kernel_config.weight_norm
        kernel_steerable = kernel_config.steerable
        kernel_init_spatial_value = kernel_config.init_spatial_value
        kernel_bias_init = kernel_config.bias_init
        kernel_input_scale = kernel_config.input_scale
        kernel_sampling_rate_norm = kernel_config.sampling_rate_norm

        # Define Convolution Type:
        # -------------------------
        # Unpack other conv_config values in case normal convolutions are used.
        conv_type = conv_config.type
        conv_horizon = conv_config.horizon
        conv_padding = conv_config.padding
        conv_stride = conv_config.stride
        conv_bias = conv_config.bias

        # Unpack mask_config
        mask_use = mask_config.use
        mask_type = mask_config.type
        mask_init_value = mask_config.init_value
        mask_temperature = mask_config.temperature
        mask_dynamic_cropping = mask_config.dynamic_cropping
        mask_threshold = mask_config.threshold

        # Define partials for types of convs
        if conv_type == "CKConv":
            ConvType = partial(
                ckconv.nn.CKConv,
                kernel_type=kernel_type,
                kernel_dim_linear=kernel_dim_linear,
                kernel_no_hidden=kernel_no_hidden,
                kernel_no_layers=kernel_no_layers,
                kernel_activ_function=kernel_activ_function,
                kernel_norm=kernel_norm,
                kernel_omega_0=kernel_omega_0,
                kernel_learn_omega_0=kernel_learn_omega_0,
                kernel_weight_norm=kernel_weight_norm,
                kernel_steerable=kernel_steerable,
                kernel_init_spatial_value=kernel_init_spatial_value,
                kernel_bias_init=kernel_bias_init,
                kernel_input_scale=kernel_input_scale,
                kernel_sampling_rate_norm=kernel_sampling_rate_norm,
                conv_horizon=conv_horizon,
                conv_padding=conv_padding,
                conv_stride=conv_stride,
                conv_bias=conv_bias,
            )
        elif conv_type == "FlexConv":
            ConvType = partial(
                ckconv.nn.FlexConv,
                kernel_type=kernel_type,
                kernel_dim_linear=kernel_dim_linear,
                kernel_no_hidden=kernel_no_hidden,
                kernel_no_layers=kernel_no_layers,
                kernel_activ_function=kernel_activ_function,
                kernel_norm=kernel_norm,
                kernel_omega_0=kernel_omega_0,
                kernel_learn_omega_0=kernel_learn_omega_0,
                kernel_weight_norm=kernel_weight_norm,
                kernel_steerable=kernel_steerable,
                kernel_init_spatial_value=kernel_init_spatial_value,
                kernel_bias_init=kernel_bias_init,
                kernel_input_scale=kernel_input_scale,
                kernel_sampling_rate_norm=kernel_sampling_rate_norm,
                conv_horizon=conv_horizon,
                conv_padding=conv_padding,
                conv_stride=conv_stride,
                conv_bias=conv_bias,
                mask_use=mask_use,
                mask_type=mask_type,
                mask_init_value=mask_init_value,
                mask_temperature=mask_temperature,
                mask_dynamic_cropping=mask_dynamic_cropping,
                mask_threshold=mask_threshold,
            )
        elif conv_type == "Conv":
            ConvType = partial(
                getattr(torch.nn, f"Conv{kernel_dim_linear}d"),
                kernel_size=int(conv_horizon),
                padding=conv_padding,
                stride=conv_stride,
                bias=conv_bias,
            )
        else:
            raise NotImplementedError(f"conv_type = {conv_type}")
        # -------------------------

        # Define NormType
        NormType = {
            "BatchNorm": getattr(torch.nn, f"BatchNorm{kernel_dim_linear}d"),
            "LayerNorm": ckconv.nn.LayerNorm,
        }[norm]

        NonlinearType = {"ReLU": torch.nn.ReLU, "LeakyReLU": torch.nn.LeakyReLU}[
            nonlinearity
        ]

        # Define LinearType
        LinearType = getattr(ckconv.nn, f"Linear{kernel_dim_linear}d")

        # Create Blocks
        # -------------------------
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]

        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )

        blocks = []
        for i in range(no_blocks):

            if i == 0:
                input_ch = in_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                TCNBlock(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    dropout=dropout,
                )
            )
            # if pool: # Pool is not used in our experiments
            #     blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])

        self.finallyr = torch.nn.Linear(
            in_features=final_no_hidden, out_features=out_channels
        )
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(
            value=0.0
        )  # TODO: Is this initialization optimal?

    def forward(self, x):
        raise NotImplementedError()


class CopyMemory_TCN(TCNBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        mask_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            net_config=net_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
            mask_config=mask_config,
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out.transpose(1, 2))
        return out


class SeqData_TCN(TCNBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        mask_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            net_config=net_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
            mask_config=mask_config,
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        return out
