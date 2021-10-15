import torch
import ckconv
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmUp_LRScheduler(torch.nn.Module):
    def __init__(
        self,
        optimizer,
        lr_scheduler,
        warmup_iterations,
    ):
        super(LinearWarmUp_LRScheduler, self).__init__()

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda iter: iter / warmup_iterations,
        )
        self.lr_scheduler = lr_scheduler
        self.warmup_iterations = warmup_iterations
        self.iteration = 0

    def step(self):
        if self.iteration <= self.warmup_iterations:
            self.warmup_scheduler.step()
        else:
            self.lr_scheduler.step()
        self.iteration += 1
