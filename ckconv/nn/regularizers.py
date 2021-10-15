import torch

import ckconv.nn.kernelnet
from ckconv.nn import CKConv, FlexConv, MultipliedLinear2d, MultipliedLinear1d


class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.

        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: CKConv,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, CKConv):
                loss += m.conv_kernel.norm(self.norm_type)
                loss += m.bias.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss


class LimitLnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the FlexConv limits in a CKCNN.

        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: CKConv,
    ):
        loss = 0.0
        # Go through modules that are instances of FlexConvs
        for m in model.modules():
            if isinstance(m, FlexConv):
                loss += m.limits.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss


class MagnitudeRegularization(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """"""
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: CKConv,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, (MultipliedLinear1d, MultipliedLinear2d)):
                loss += (m.omega_0 * m.weight).norm(self.norm_type)
            # elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
            #     loss += m.weight.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss


class SmoothnessRegularizer(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """"""
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: torch.nn.Module,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, (ckconv.nn.kernelnet.GaborLayer)):
                loss += (m.linear.weight).norm(self.norm_type)
            # elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
            #     loss += m.weight.norm(self.norm_type)

        loss = self.weight_loss * loss
        return loss
