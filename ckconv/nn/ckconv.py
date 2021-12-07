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
        horizon: int,
        kernel_type = "MAGNet",
        kernel_dim_linear = 2,
        kernel_no_hidden = 32,
        kernel_no_layers = 3,
        kernel_activ_function = "ReLU",
        kernel_norm = "BatchNorm",
        kernel_omega_0 = 0.0,
        kernel_learn_omega_0 = False,
        kernel_weight_norm = False,
        kernel_steerable = False,
        kernel_init_spatial_value = 1.0,
        kernel_bias_init = None,
        kernel_input_scale = 25.6,
        kernel_sampling_rate_norm = 1.0,
        conv_use_fft = False,
        conv_bias = True,
        conv_padding = "same",
        conv_stride = 1,
    ):
        """
        Continuous Kernel Convolution.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param horizon: Maximum kernel size. Recommended to be odd and cover the entire image.
        :param kernel_type: Identifier for the type of kernel generator to use.
        :param kernel_dim_linear: Dimensionality of the input signal, e.g. 2 for images.
        :param kernel_no_hidden: Amount of hidden channels to use.
        :param kernel_activ_function: Activation function for type=MLP.
        :param kernel_norm: Normalization function for type=MLP.
        :param kernel_weight_norm: Weight normalization, for type=[MLP, SIREN, nSIREN].
        :param kernel_no_layers: Amount of layers to use in kernel generator.
        :param kernel_omega_0: Initial value for omega_0, for type=SIREN.
        :param kernel_learn_omega_0: Whether to learn omega_0, for type=SIREN.
        :param kernel_steerable: Whether to learn steerable kernels, for type=MAGNet.
        :param kernel_init_spatial_value: Initial mu for gabor filters, for type=[GaborNet, MAGNet].
        :param kernel_bias_init: Bias init strategy, for all types but type=MLP.
        :param kernel_input_scale: Scaling factor for linear functions, for type=[GaborNet, MAGNet].
        :param kernel_sampling_rate_norm: Kernel scaling factor for sampling rate normalization.
        :param conv_use_fft: Whether to use FFT implementation of convolution.
        :param conv_bias: Whether to use bias in kernel generator. TODO(rjbruin): move to kernel_config.
        :param conv_padding: Padding strategy for convolution.
        :param conv_stride: Stride applied in convolution.
        """

        super().__init__()

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
                hidden_channels=kernel_no_hidden,
                activation_function=kernel_activ_function,
                norm_type=kernel_norm,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=conv_bias,
            )
        if kernel_type == "SIREN":
            self.Kernel = ckconv.nn.ck.SIREN(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=conv_bias,
                bias_init=kernel_bias_init,
                omega_0=kernel_omega_0,
                learn_omega_0=kernel_learn_omega_0,
            )
        elif kernel_type == "nSIREN":
            self.Kernel = ckconv.nn.ck.nSIREN(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                weight_norm=kernel_weight_norm,
                no_layers=kernel_no_layers,
                bias=conv_bias,
                bias_init=kernel_bias_init,
                omega_0=kernel_omega_0,
                learn_omega_0=kernel_learn_omega_0,
            )
        elif kernel_type == "Fourier":
            self.Kernel = ckconv.nn.ck.FourierNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                bias=conv_bias,
                bias_init=kernel_bias_init,
            )
        elif kernel_type == "Gabor":
            self.Kernel = ckconv.nn.ck.GaborNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                bias=conv_bias,
                bias_init=kernel_bias_init,
                init_spatial_value=kernel_init_spatial_value,
                input_scale=kernel_input_scale,
            )
        elif kernel_type == "MAGNet":
            self.Kernel = ckconv.nn.ck.MAGNet(
                dim_linear=kernel_dim_linear,
                out_channels=out_channels * in_channels,
                hidden_channels=kernel_no_hidden,
                no_layers=kernel_no_layers,
                steerable=kernel_steerable,
                bias=conv_bias,
                bias_init=kernel_bias_init,
                init_spatial_value=kernel_init_spatial_value,
                input_scale=kernel_input_scale,
            )

        if conv_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        # Save arguments in self
        # ---------------------
        # Non-persistent values
        self.padding = conv_padding
        self.stride = conv_stride
        self.rel_positions = None
        self.kernel_dim_linear = kernel_dim_linear
        self.horizon = horizon
        self.use_fftconv = conv_use_fft
        self.kernel_sampling_rate_norm = kernel_sampling_rate_norm

        # Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("conv_kernel", torch.zeros(in_channels), persistent=False)

        # Define convolution type
        conv_type = "conv"
        if conv_use_fft:
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
        horizon: int,
        kernel_type = "MAGNet",
        kernel_dim_linear = 2,
        kernel_no_hidden = 32,
        kernel_no_layers = 3,
        kernel_activ_function = "ReLU",
        kernel_norm = "BatchNorm",
        kernel_omega_0 = 0.0,
        kernel_learn_omega_0 = False,
        kernel_weight_norm = False,
        kernel_steerable = False,
        kernel_init_spatial_value = 1.0,
        kernel_bias_init = None,
        kernel_input_scale = 25.6,
        kernel_sampling_rate_norm = 1.0,
        conv_use_fft = False,
        conv_bias = True,
        conv_padding = "same",
        conv_stride = 1,
        mask_use = True,
        mask_type = "gaussian",
        mask_init_value = 0.075,
        mask_temperature = 15.0,
        mask_dynamic_cropping = True,
        mask_threshold = 0.1,
    ):
        """
        Flexible Size Continuous Kernel Convolution.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param horizon: Maximum kernel size. Recommended to be odd and cover the entire image.
        :param kernel_type: Identifier for the type of kernel generator to use.
        :param kernel_dim_linear: Dimensionality of the input signal, e.g. 2 for images.
        :param kernel_no_hidden: Amount of hidden channels to use.
        :param kernel_activ_function: Activation function for type=MLP.
        :param kernel_norm: Normalization function for type=MLP.
        :param kernel_weight_norm: Weight normalization, for type=[MLP, SIREN, nSIREN].
        :param kernel_no_layers: Amount of layers to use in kernel generator.
        :param kernel_omega_0: Initial value for omega_0, for type=SIREN.
        :param kernel_learn_omega_0: Whether to learn omega_0, for type=SIREN.
        :param kernel_steerable: Whether to learn steerable kernels, for type=MAGNet.
        :param kernel_init_spatial_value: Initial mu for gabor filters, for type=[GaborNet, MAGNet].
        :param kernel_bias_init: Bias init strategy, for all types but type=MLP.
        :param kernel_input_scale: Scaling factor for linear functions, for type=[GaborNet, MAGNet].
        :param kernel_sampling_rate_norm: Kernel scaling factor for sampling rate normalization.
        :param conv_use_fft: Whether to use FFT implementation of convolution.
        :param conv_bias: Whether to use bias in kernel generator. TODO(rjbruin): move to kernel_config.
        :param conv_padding: Padding strategy for convolution.
        :param conv_stride: Stride applied in convolution.
        :param mask_use: Whether to apply Gaussian mask.
        :param mask_type: Type of mask. Recommended to use "gaussian".
        :param mask_init_value: Initial value for the size of the kernel.
        :param mask_temperature: Temperature of the sigmoid function, for type=sigmoid.
        :param mask_dynamic_cropping: Whether to crop away pixels below the threshold.
        :param mask_threshold: Threshold for cropping pixels. Recommended to be 15.0.
        """

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
            kernel_init_spatial_value = init_spatial_value

        # Super
        super().__init__(
            in_channels,
            out_channels,
            horizon,
            kernel_type,
            kernel_dim_linear,
            kernel_no_hidden,
            kernel_no_layers,
            kernel_activ_function,
            kernel_norm,
            kernel_omega_0,
            kernel_learn_omega_0,
            kernel_weight_norm,
            kernel_steerable,
            kernel_init_spatial_value,
            kernel_bias_init,
            kernel_input_scale,
            kernel_sampling_rate_norm,
            conv_use_fft,
            conv_bias,
            conv_padding,
            conv_stride,
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
