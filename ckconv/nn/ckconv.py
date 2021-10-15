import copy
import math
from torch.profiler import record_function

import torch
import torch.fft
import torch.nn
import ckconv
import ckconv.nn.functional as ckconv_F
from ckconv.utils.grids import rel_positions_grid

# typing
from omegaconf import OmegaConf


class CKConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        """
        Creates a Continuous Kernel Convolution.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param hidden_channels: Number of hidden units in the network parameterizing the ConvKernel (KernelNet).
        :param activation_function: Activation function used in KernelNet.
        :param norm_type: Normalization type used in KernelNet. (only for non-Sine KernelNets).
        :param dim_linear: patial dimension of the input, e.g., for audio = 1, images = 2 (only 1 suported).
        :param bias: If True, adds a learnable bias to the output.
        :param omega_0: Value of the omega_0 value of the KernelNet. (only for non-Sine KernelNets).
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        :param sampling_rate_norm: Normalization factor for deploying at a different sampling rate than trained.
        """

        super().__init__()

        # Unpack values from kernel_config
        kernel_type = kernel_config.type
        kernel_dim_linear = kernel_config.dim_linear
        kernel_hidden_channels = kernel_config.no_hidden
        kernel_activ_function = kernel_config.activ_function
        kernel_norm = kernel_config.norm
        kernel_weight_norm = kernel_config.weight_norm
        kernel_no_layers = kernel_config.no_layers
        kernel_omega_0 = kernel_config.omega_0
        kernel_learn_omega_0 = kernel_config.learn_omega_0
        kernel_steerable = kernel_config.steerable
        kernel_init_spatial_value = kernel_config.init_spatial_value
        kernel_bias_init = kernel_config.bias_init
        kernel_input_scale = kernel_config.input_scale
        kernel_sampling_rate_norm = kernel_config.sampling_rate_norm

        # Unpack values from conv_config
        use_fftconv = conv_config.use_fft
        horizon = conv_config.horizon
        bias = conv_config.bias
        padding = conv_config.padding
        stride = conv_config.stride

        # Since kernels are defined between [-1, 1] if values are bigger than one, they are modified.
        if kernel_init_spatial_value > 1.0:
            kernel_init_spatial_value = 1.0
            print(
                f"Received kernel_init_spatial is bigger than one, and has been set to 1.0."
                f"Current value: {kernel_init_spatial_value}"
            )

        # Create the kernel
        if kernel_type == "MLP":
            self.Kernel = ckconv.nn.ck.MLP(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                activation_function=kernel_activ_function,
                norm_type=kernel_norm,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=bias,
            )
        if kernel_type == "SIREN":
            self.Kernel = ckconv.nn.ck.SIREN(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=bias,
                bias_init=kernel_bias_init,
                omega_0=kernel_omega_0,
                learn_omega_0=kernel_learn_omega_0,
            )
        elif kernel_type == "nSIREN":
            self.Kernel = ckconv.nn.ck.nSIREN(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=bias,
                bias_init=kernel_bias_init,
                omega_0=kernel_omega_0,
                learn_omega_0=kernel_learn_omega_0,
            )
        elif kernel_type == "Fourier":
            self.Kernel = ckconv.nn.ck.FourierNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                no_layers=kernel_no_layers,
                bias=bias,
                bias_init=kernel_bias_init,
            )
        elif kernel_type == "Gabor":
            self.Kernel = ckconv.nn.ck.GaborNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                no_layers=kernel_no_layers,
                bias=bias,
                bias_init=kernel_bias_init,
                init_spatial_value=kernel_init_spatial_value,
                input_scale=kernel_input_scale,
            )
        elif kernel_type == "MAGNet":
            self.Kernel = ckconv.nn.ck.MAGNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_hidden_channels,
                no_layers=kernel_no_layers,
                steerable=kernel_steerable,
                bias=bias,
                bias_init=kernel_bias_init,
                init_spatial_value=kernel_init_spatial_value,
                input_scale=kernel_input_scale,
            )

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        # Save arguments in self
        # ---------------------
        # Non-persistent values
        self.padding = padding
        self.stride = stride
        self.rel_positions = None
        self.kernel_dim_linear = kernel_dim_linear
        self.horizon = horizon
        self.use_fftconv = use_fftconv
        self.kernel_sampling_rate_norm = kernel_sampling_rate_norm

        # Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("conv_kernel", torch.zeros(in_channels), persistent=False)

        # Define convolution type
        conv_type = "conv"
        if use_fftconv:
            conv_type = "fft" + conv_type
        if kernel_dim_linear == 1:
            conv_type = "causal_" + conv_type
        self.conv = getattr(ckconv_F, conv_type)

    def forward(self, x):
        # Construct kernel
        x_shape = x.shape

        rel_pos = self.handle_rel_positions(x)
        conv_kernel = self.Kernel(rel_pos).view(-1, x_shape[1], *rel_pos.shape[2:])
        conv_kernel *= self.kernel_sampling_rate_norm

        # For computation of "weight_decay"
        self.conv_kernel = conv_kernel

        return self.conv(x, conv_kernel, self.bias)

    def handle_rel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if self.rel_positions is None:
            if self.train_length[0] == 0:

                # Decide the extend of the rel_positions vector
                if self.horizon == "full":
                    self.train_length[0] = (2 * x.shape[-1]) - 1
                elif self.horizon == "same":
                    self.train_length[0] = x.shape[-1]
                elif int(self.horizon) % 2 == 1:
                    # Odd number
                    self.train_length[0] = int(self.horizon)
                else:
                    raise ValueError(
                        f"The horizon argument of the operation must be either 'full', 'same' or an odd number in string format. Current value: {self.horizon}"
                    )

            # Creates the vector of relative positions.
            rel_positions = rel_positions_grid(
                grid_sizes=self.train_length.repeat(self.kernel_dim_linear)
            ).unsqueeze(0)
            self.rel_positions = rel_positions.to(x.device)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

        return self.rel_positions


class FlexConv(CKConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        mask_config: OmegaConf,
    ):
        """"""

        # Unpack mask_config values:
        mask_use = mask_config.use
        mask_type = mask_config.type
        mask_init_value = mask_config.init_value
        mask_temperature = mask_config.temperature
        mask_dynamic_cropping = mask_config.dynamic_cropping
        mask_threshold = mask_config.threshold

        # conv_cache = conv_config.cache

        """
        Initialise init_spatial_value:
        ------------------------------
        It defines the extend on the input space on which the kernel will be initialized, e.g., mu in GaborNet. Since
         mask_init_value defines the variance of the kernel, it must be defined in a larger (apprx. 1.667 * mask_init_value)
         region than the variance.
        """

        if mask_type == "gaussian":
            init_spatial_value = (
                mask_init_value * 1.667
            )  # TODO: Might be different for different tasks
        elif mask_type == "sigmoid":
            init_spatial_value = mask_init_value
        else:
            raise NotImplementedError()

        # Modify the kernel_config if required
        if init_spatial_value != mask_init_value:
            kernel_config.init_spatial_value = init_spatial_value

        # Super
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Define convolution types
        # Get names:
        conv_types = {
            "spatial": "conv",
            "fft": "fftconv",
        }
        # If 1D, then use causal convolutions:
        if self.kernel_dim_linear == 1:
            for (key, value) in conv_types.items():
                conv_types[key] = "causal_" + value
        # Save convolution functions in self:
        for (key, value) in conv_types.items():
            conv_types[key] = getattr(ckconv_F, value)
        self.conv_types = conv_types

        # Define mask constructor
        if mask_type == "sigmoid":
            raise NotImplementedError(
                f"mask_type {mask_type} not currently working."
            )  # TODO: we need to compute roots analytically
        self.mask_constructor = globals()[f"{mask_type}_mask_{self.kernel_dim_linear}d"]

        # Define the parameters of the mask
        mask_params = {
            "gaussian": {
                1: torch.Tensor(
                    [[1.0, mask_init_value]]
                ),  # TODO: How to initialize this properly is not entirely clear yet.
                2: torch.Tensor([[0.0, mask_init_value], [0.0, mask_init_value]]),
            },
            "sigmoid": {
                1: torch.Tensor([[-mask_init_value, mask_init_value]]),
                2: torch.Tensor(
                    [
                        [-mask_init_value, mask_init_value],
                        [-mask_init_value, mask_init_value],
                    ]
                ),
            },
        }[mask_type][
            self.kernel_dim_linear
        ]  # TODO: Sigmoid mask not checked in 1D yet
        self.mask_params = torch.nn.Parameter(mask_params)

        # Define temperature
        temperature = mask_temperature * torch.ones(1)
        self.register_buffer("temperature", temperature, persistent=True)

        # Define threshold of mask for dynamic cropping
        mask_threshold = mask_threshold * torch.ones(1)
        self.register_buffer("mask_threshold", mask_threshold, persistent=True)

        # Save values in self
        self.mask_dynamic_cropping = mask_dynamic_cropping
        self.mask = mask_use

        # Kernel caching
        # DEBUG(rjbruin): cache causes problems with multi-GPU behavior
        # self.conv_cache = conv_cache
        # self.register_buffer('conv_kernel', None, persistent=False)

    def forward(self, x):
        # if not self.conv_cache or (self.training or self.conv_kernel is None):
        with record_function("flexconv"):
            # Construct kernel
            x_shape = x.shape

            with record_function("handle_rel_positions"):
                rel_pos = self.handle_rel_positions(x)

            # Dynamic cropping
            if self.mask_dynamic_cropping:
                with record_function("dynamic_cropping"):
                    # Based on the current mean and sigma values, compute the [min, max] values of the array.
                    with torch.no_grad():
                        if self.kernel_dim_linear == 1:
                            # Find root
                            x_root = gauss_min_root(
                                self.mask_threshold,
                                self.mask_params[0, 0],
                                self.mask_params[0, 1],
                            )
                            # Only if the root is within [-1, 1], cropping must me made. Otherwise, the same grid is preserved.
                            if abs(x_root.item()) < 1.0:
                                x_grid = crop_relative_positions_1d(
                                    rel_pos[0, 0, :], x_root
                                )
                                rel_pos = torch.stack(
                                    torch.meshgrid(x_grid), dim=0
                                ).unsqueeze(0)

                        elif self.kernel_dim_linear == 2:
                            # Find roots
                            abs_x_root = gauss_max_absolute_root(
                                self.mask_threshold,
                                self.mask_params[0, 0],
                                self.mask_params[0, 1],
                            )
                            abs_y_root = gauss_max_absolute_root(
                                self.mask_threshold,
                                self.mask_params[1, 0],
                                self.mask_params[1, 1],
                            )
                            # Only if one of the roots is within [-1, 1], cropping must me made. Otherwise, the same grid is preserved.
                            if abs_x_root.item() < 1.0 or abs_y_root.item() < 1.0:
                                x_grid = crop_relative_positions_2d(
                                    rel_pos[0, 1, 0, :], abs_x_root
                                )
                                y_grid = crop_relative_positions_2d(
                                    rel_pos[0, 0, :, 0], abs_y_root
                                )
                                rel_pos = torch.stack(
                                    torch.meshgrid(x_grid, y_grid), dim=0
                                ).unsqueeze(0)

            with record_function("kernel"):
                # KernelNet
                conv_kernel = self.Kernel(rel_pos).view(
                    -1, x_shape[1], *rel_pos.shape[2:]
                )

                # Sampling rate norm
                conv_kernel *= self.kernel_sampling_rate_norm

            if self.mask:
                with record_function("mask"):
                    # Construct mask and multiply conv-kernel with it.
                    mask = self.mask_constructor(rel_pos, self.mask_params).view(
                        1, 1, *rel_pos.shape[2:]
                    )
                    if self.mask_constructor in [sigmoid_mask1d, sigmoid_mask2d]:
                        mask = mask ** 2

                    # For computation of "weight_decay"
                    self.conv_kernel = mask * conv_kernel
            else:
                self.conv_kernel = conv_kernel

        with record_function("convolution"):
            # Convolution
            # See if the kernel is big enough to use fft_conv.
            # NOTE(rjbruin): changed this to measure against `self.conv_kernel`
            # instead of `conv_kernel`, which means now whether to use FFT or
            # not is dependent on the **masked kernel**, not the **unmasked
            # kernel**.
            size = torch.tensor(self.conv_kernel.shape[2:])
            if self.use_fftconv and (
                False not in (size > 50)
            ):  # TODO haven't checked if 2D fft works with FlexConv. It doess with CKConv, though.
                # If the kernel is larger than 50, use fftconv
                out = self.conv_types["fft"](x, self.conv_kernel, self.bias)
            else:
                out = self.conv_types["spatial"](x, self.conv_kernel, self.bias)

        return out


###############################
# Gaussian Masks / Operations #
###############################


def gaussian_mask_2d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    mask_y = gaussian_1d(rel_positions[0, 0], mask_params[0, 0], mask_params[0, 1])
    mask_x = gaussian_1d(rel_positions[0, 1], mask_params[1, 0], mask_params[1, 1])
    return mask_y * mask_x


def gaussian_mask_1d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return gaussian_1d(rel_positions[0, 0], mask_params[0, 0], mask_params[0, 1])


def gaussian_1d(
    x: torch.Tensor,
    mean: float,
    sigma: float,
) -> torch.Tensor:
    return torch.exp(-1 / 2 * ((1 / sigma) * (x - mean)) ** 2)


def crop_relative_positions_1d(
    dim_i_rel_positions: torch.Tensor,
    root: float,
):
    # In 1D, only one part of the array must be cut.
    if abs(root) >= 1.0:
        return dim_i_rel_positions
    else:
        # Based on the root, take the negative values of the array and find the first index that becomes one
        index = torch.nonzero(root < dim_i_rel_positions)[0]
        return dim_i_rel_positions[index:]


def crop_relative_positions_2d(
    dim_i_rel_positions: torch.Tensor,
    root: float,
):
    # If root is larger than one, then all values must be kept
    if root >= 1.0:
        return dim_i_rel_positions
    else:
        # Based on the root, take the negative values of the array and find the first index that becomes one
        index = torch.nonzero(
            -root < dim_i_rel_positions[: (dim_i_rel_positions.shape[-1] // 2) + 1]
        )[0]
        return dim_i_rel_positions[index : dim_i_rel_positions.shape[-1] - index]


def gauss_min_root(
    thresh: float,
    mean: float,
    sigma: float,
):
    root1, root2 = gaussian_inv_thresh(thresh, mean, sigma)
    if root1 < root2:
        return root1
    else:
        return root2


def gauss_max_absolute_root(
    thresh: float,
    mean: float,
    sigma: float,
):
    root1, root2 = gaussian_inv_thresh(thresh, mean, sigma)
    root1, root2 = torch.abs(root1), torch.abs(root2)
    # Then compare them and return the highest of both
    if root1 > root2:
        return root1
    else:
        return root2


def gaussian_inv_thresh(
    thresh: float,
    mean: float,
    sigma: float,
):
    # Based on the treshold value, compute the absolute value of the roots
    aux = sigma * torch.sqrt(-2 * torch.log(thresh))
    root1 = mean - aux
    root2 = mean + aux
    return root1, root2


##############################
# Sigmoid Masks / Operations #
##############################


def sigmoid_mask2d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    temperature = temperature.item()

    mask_y = temperature_sigmoid(
        rel_positions[0, 0], mask_params[0, 0], temperature
    ) - temperature_sigmoid(rel_positions[0, 0], mask_params[0, 1], temperature)
    mask_x = temperature_sigmoid(
        rel_positions[0, 1], mask_params[1, 0], temperature
    ) - temperature_sigmoid(rel_positions[0, 1], mask_params[1, 1], temperature)
    return mask_y * mask_x


def sigmoid_mask1d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    temperature = temperature.item()
    return temperature_sigmoid(
        rel_positions[0, 0], mask_params[0, 0], temperature
    ) - temperature_sigmoid(rel_positions[0, 0], mask_params[0, 1], temperature)


def temperature_sigmoid(x, offset, temperature):
    return torch.sigmoid(temperature * (x - offset))
    # Weirdly enough, the line below produces nan values for large limits.
    # return 1.0 / (1.0 + torch.exp(-(x - offset) * temperature))
