import torch
import torch.nn.functional as F
import torch.fft

from typing import Tuple, Optional


def conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (Optional, Tensor) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    # Input tensors are assumed to have dimensionality [batch_size, no_channels, spatial_dim1, .., spatial_dimN].
    spatial_dim = len(x.shape) - 2
    if spatial_dim == 1:
        conv_function = torch.nn.functional.conv1d
    elif spatial_dim == 2:
        conv_function = torch.nn.functional.conv2d
    elif spatial_dim == 3:
        conv_function = torch.nn.functional.conv3d
    else:
        raise NotImplementedError(
            "Convolution is not implemented for inputs of spatial dimension {}".format(
                spatial_dim
            )
        )

    if (kernel.shape[-2] % 2 == 0) or (
        kernel.shape[-1] % 2 == 0
    ):  # TODO This is a problem! We add
        raise AttributeError(
            "Convolutional kernels must have odd dimensionality. Received: ({}, {})".format(
                kernel.shape[-2], kernel.shape[-1]
            )
        )

    padding = (
        int((kernel.shape[-2] - 1) / 2),
        int((kernel.shape[-1] - 1) / 2),
    )

    return conv_function(x, kernel, bias=bias, padding=padding, stride=1)


def fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    double_precision: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    # Input tensors are assumed to have dimensionality [batch_size, no_channels, spatial_dim1, .., spatial_dimN].
    x_shape = x.shape
    spatial_dim = len(x.shape) - 2

    if (kernel.shape[-2] % 2 == 0) or (
        kernel.shape[-1] % 2 == 0
    ):
        raise AttributeError(
            "Convolutional kernels must have odd dimensionality. Received: ({}, {})".format(
                kernel.shape[-2], kernel.shape[-1]
            )
        )

    # 1. Pad the input and the kernel to make them equally big. Required for fft.
    # -------------------------------
    # x:
    padding_x = int((kernel.shape[-1] - 1) / 2)
    padding_x = (2 * spatial_dim) * [
        padding_x
    ]  # Creates a vector [padding_left_dim1, padding_right_dim1, ..., padding_left_dimN, padding_right_dimN]
    x = F.pad(x, padding_x)
    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, [0, 1])

    # kernel:
    padding_kernel = [
        pad
        for i in reversed(range(2, x.ndim))
        for pad in [0, x.shape[i] - kernel.shape[i]]
    ]
    kernel = F.pad(kernel, padding_kernel)
    # -------------------------------

    # 3. Perform fourier transform
    if double_precision:
        # We can make usage of double precision to make more accurate approximations of the convolution response.
        x = x.double()
        kernel = kernel.double()

    x_fr = torch.fft.rfftn(x, dim=tuple(range(2, x.ndim)))
    kernel_fr = torch.fft.rfftn(kernel, dim=tuple(range(2, kernel.ndim)))

    # 4. Multiply the transformed matrices:
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    kernel_fr = torch.conj(kernel_fr)
    output_fr = torch.einsum("bi..., oi... -> bo...", x_fr, kernel_fr)
    # output_fr = (x_fr.unsqueeze(1) * kernel_fr.unsqueeze(0)).sum(
    #     2
    # )  # 'ab..., cb... -> ac...'

    # 5. Compute inverse FFT, and remove extra padded values
    # Once we are back in the spatial domain, we can go back to float precision, if double used.
    out = torch.fft.irfftn(output_fr, dim=tuple(range(2, x.ndim))).float()

    # No PyTorch function exists for cropping, it seems
    if spatial_dim == 1:
        out = out[:, :, : x_shape[-1]]
    elif spatial_dim == 2:
        out = out[:, :, : x_shape[-2], : x_shape[-1]]
    elif spatial_dim == 3:
        out = out[:, :, : x_shape[-3], : x_shape[-2], : x_shape[-1]]

    # 6. Optionally, add a bias term before returning.
    if bias is not None:
        reshape_dim = spatial_dim * [1]
        out = out + bias.view(1, -1, *reshape_dim)

    return out
