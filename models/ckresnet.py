import torch
import ckconv.nn
from functools import partial
from .residual_block import ResidualBlockBase

# typing
from omegaconf import OmegaConf
from typing import Tuple, Union
from ckconv.nn import FlexConv, CKConv
from torch.nn import Conv1d, Conv2d
from srf.nn import Srf_layer_shared


class ResNetBlock(ResidualBlockBase):
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
        Creates a Residual Block as in the original ResNet paper (He et al., 2016)

        input
         | ---------------|
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         |                |
         CKConv           |
         LayerNorm        |
         |                |
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
            NormType=NormType,
            NonlinearType=NonlinearType,
            LinearType=LinearType,
            dropout=dropout,
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        # Following Sosnovik et al. 2020, dropout placed after first ReLU.
        out = self.dp(self.nonlinear(self.norm1(self.cconv1(x))))
        out = self.nonlinear(self.norm2(self.cconv2(out)) + shortcut)
        return out


class Img_ResNet(torch.nn.Module):
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

        # Unpack dim_linear
        kernel_scale_sigma = kernel_config.srf.scale

        # Unpack conv_type
        conv_type = conv_config.type

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
        conv_use_fft = conv_config.use_fft
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
                horizon=conv_horizon,
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
                conv_use_fft=conv_use_fft,
                conv_bias=conv_bias,
                conv_padding=conv_padding,
                conv_stride=1,
            )
        elif conv_type == "FlexConv":
            ConvType = partial(
                ckconv.nn.FlexConv,
                horizon=conv_horizon,
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
                conv_use_fft=conv_use_fft,
                conv_bias=conv_bias,
                conv_padding=conv_padding,
                conv_stride=conv_stride,
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
        elif conv_type == "SRF":
            ConvType = partial(
                Srf_layer_shared,
                init_k=2.0,
                init_order=4.0,
                init_scale=0.0,
                use_cuda=True,  # NOTE(rjbruin): hardcoded for now
                scale_sigma=kernel_scale_sigma,
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

        # Create Input Layers
        self.cconv1 = ConvType(in_channels=in_channels, out_channels=hidden_channels)
        self.norm1 = NormType(hidden_channels)
        self.nonlinear = NonlinearType()

        # Create Blocks
        # -------------------------
        # Create vector of width_factors:
        # If value is zero, then all values are one
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
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                ResNetBlock(
                    input_ch,
                    hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    dropout=dropout,
                )
            )
            # if pool: # Pool is not used in our experiments
            #     raise NotImplementedError()
            #     # blocks.append(torch.nn.MaxPool1d(kernel_size=2))

        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # instantiate last layer
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        # Initialize finallyr
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------

        # Save variables in self
        self.dim_linear = kernel_dim_linear

    def forward(self, x):
        # First layers
        out = self.nonlinear(self.norm1(self.cconv1(x)))
        # Blocks
        out = self.blocks(out)
        # Pool
        out = torch.nn.functional.adaptive_avg_pool2d(
            out,
            (1,) * self.dim_linear,
        )
        # Final layer
        out = self.out_layer(out)
        return out.squeeze()


class SeqData_ResNet(Img_ResNet):
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
        # First layers
        out = torch.relu(self.norm1(self.cconv1(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_layer(out[:, :, -1:])
        return out.squeeze(-1)


# class Segmentation_ResNet(ResNet):  # TODO!
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         hidden_channels: int,
#         num_blocks: int,
#         kernelnet_hidden_channels: int,
#         kernelnet_hidden_bottleneck_factor: float,
#         kernelnet_activation_function: str,
#         kernelnet_norm_type: str,
#         kernelnet_weight_norm: bool,
#         kernelnet_no_layers: int,
#         dim_linear: int,
#         bias: bool,
#         omega_0: float,
#         learn_omega_0: bool,
#         omega_0_decay: str,
#         ckconv_horizon: str,
#         dropout: float,
#         weight_dropout: float,
#         sampling_rate_norm: float,
#         pool: bool,
#         conv_type: str,
#         omega_0_scheme: str,
#         omega_0_hidden: float,
#         norm: str,
#         dynamic_cropping: bool,
#         mask_temperature: float,
#         mask_type: str,
#         use_fftconv: bool,
#         kernel_type: str,
#         bottleneck_layers: int,
#         mask_init_value: float,
#         block_width_factors: Tuple[float],
#         time_flexconv: bool,
#     ):
#         # Downsampling branch
#         backbone_in = in_channels
#         self.bottleneck = False
#         if bottleneck_layers > 0:
#             backbone_in = hidden_channels
#             self.bottleneck = True
#
#         super().__init__(
#             backbone_in,
#             hidden_channels,
#             num_blocks,
#             kernelnet_hidden_channels,
#             kernelnet_hidden_bottleneck_factor,
#             kernelnet_activation_function,
#             kernelnet_norm_type,
#             kernelnet_weight_norm,
#             kernelnet_no_layers,
#             dim_linear,
#             bias,
#             omega_0,
#             learn_omega_0,
#             omega_0_decay,
#             ckconv_horizon,
#             dropout,
#             weight_dropout,
#             sampling_rate_norm,
#             pool,
#             conv_type,
#             omega_0_scheme,
#             omega_0_hidden,
#             norm,
#             dynamic_cropping,
#             mask_temperature,
#             mask_type,
#             use_fftconv,
#             kernel_type,
#             mask_init_value,
#             block_width_factors,
#             time_flexconv,
#         )
#
#         # calculate output channels of blocks
#         if block_width_factors[0] == 0.0:
#             final_no_hidden = hidden_channels
#         else:
#             final_no_hidden = int(hidden_channels * block_width_factors[-2])
#
#         # Backbone layers: convolution and deconvolution
#         self.down = []
#         self.up = []
#         if bottleneck_layers >= 1:
#             self.down.append(
#                 torch.nn.Conv2d(
#                     in_channels, hidden_channels, kernel_size=3, stride=2, padding=1
#                 )
#             )
#             self.up.append(
#                 torch.nn.ConvTranspose2d(
#                     final_no_hidden,
#                     final_no_hidden,
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     output_padding=1,
#                 )
#             )
#         for _ in range(1, bottleneck_layers):
#             self.down.append(
#                 torch.nn.Conv2d(
#                     hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1
#                 )
#             )
#             self.up.append(
#                 torch.nn.ConvTranspose2d(
#                     final_no_hidden,
#                     final_no_hidden,
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                     output_padding=1,
#                 )
#             )
#
#         self.down = torch.nn.Sequential(*self.down)
#         self.up = torch.nn.Sequential(*self.up)
#
#         self.finallyr = torch.nn.Conv2d(
#             final_no_hidden,
#             out_channels,
#             kernel_size=1,
#         )
#
#         self.fftconv_mode = True
#
#     def fft_on(self):
#         self.fftconv_mode = True
#         for m in self.modules():
#             if isinstance(m, ckconv.nn.CKConv):
#                 m.use_fftconv = True
#                 # DEBUG
#                 print(m, m.use_fftconv)
#
#     def fft_off(self):
#         self.fftconv_mode = False
#         for m in self.modules():
#             if isinstance(m, ckconv.nn.CKConv):
#                 m.use_fftconv = False
#                 # DEBUG
#                 print(m, m.use_fftconv)
#
#     def forward(self, x):
#         if self.bottleneck:
#             x = self.down(x)
#
#         out = self.blocks(x)
#
#         if self.bottleneck:
#             out = self.up(out)
#
#         # pool before linear
#         # out = torch.amax(out, dim=(-2, -1), keepdim=True)
#         # out = torch.nn.functional.adaptive_avg_pool2d(out, [1, 1])
#         # out = self.finallyr(out).view(x.shape[0], -1)
#         # out = torch.max_pool2d(out, kernel_size=4)
#
#         out = self.finallyr(out)
#
#         return out
