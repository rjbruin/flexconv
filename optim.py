import torch
import ckconv
import math

# typing
from omegaconf import OmegaConf


DATASET_SIZES = {
    "SpeechCommands": 24482,
    "MNIST": 60000,
    "sMNIST": 60000,
    "CIFAR10": 50000,
    "sCIFAR10": 50000,
    "CIFAR100": 50000,
    "STL10": 5000,
    "Cityscapes": 2975,
    "VOC": 1464,
    "Imagenet": 1281167,
    "Imagenet64": 1281167,
    "Imagenet32": 1281167,
    "Imagenet16": 1281167,
    "Imagenet8": 1281167,
}

CLASSES_DATASET = {
    "Imagenet": 1000,
    "Imagenet64": 1000,
    "Imagenet32": 1000,
    "Imagenet16": 1000,
    "Imagenet8": 1000,
    "CIFAR100": 10,
    "Cityscapes": 19,
    "VOC": 21,
}


def construct_optimizer(
    model: torch.nn.Module,
    cfg: OmegaConf,
):
    """
    Constructs an optimizer for a given model
    :param model: a list of parameters to be trained
    :param cfg:

    :return: optimizer
    """

    # Unpack values from cfg.train
    optimizer_type = cfg.train.optimizer
    lr = cfg.train.lr
    omega_0_lr_factor = cfg.train.omega_0_lr_factor
    mask_params_lr_factor = cfg.train.mask_params_lr_factor

    # Unpack values from cfg.train.optimizer_params
    momentum = cfg.train.optimizer_params.momentum
    nesterov = cfg.train.optimizer_params.nesterov

    # Divide params in omega_0s and other
    all_parameters = set(model.parameters())
    # omega_0s
    omega_0s = []
    for m in model.modules():
        if isinstance(
            m,
            (
                ckconv.nn.MultipliedLinear1d,
                ckconv.nn.MultipliedLinear2d,
                ckconv.nn.ck.SIRENlayer1d,
                ckconv.nn.ck.SIRENlayer2d,
            ),
        ):
            omega_0s += list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: "omega_0" in kv[0], m.named_parameters())),
                )
            )
    omega_0s = set(omega_0s)
    other_params = all_parameters - omega_0s
    # mask_params
    mask_params = []
    for m in model.modules():
        if isinstance(m, ckconv.nn.FlexConv):
            mask_params += list(
                map(
                    lambda x: x[1],
                    list(
                        filter(lambda kv: "mask_params" in kv[0], m.named_parameters())
                    ),
                )
            )
    mask_params = set(mask_params)
    other_params = other_params - mask_params
    # as list
    omega_0s = list(omega_0s)
    mask_params = list(mask_params)
    other_params = list(other_params)

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            [
                {"params": other_params},
                {"params": omega_0s, "lr": omega_0_lr_factor * lr},
                {"params": mask_params, "lr": mask_params_lr_factor * lr},
            ],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": other_params},
                {"params": omega_0s, "lr": omega_0_lr_factor * lr},
                {"params": mask_params, "lr": mask_params_lr_factor * lr},
            ],
            lr=lr,
        )
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            # weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unexpected value for type of optimizer (cfg.train.optimizer): {optimizer_type}"
        )

    return optimizer


def construct_scheduler(
    optimizer,
    cfg: OmegaConf,
):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """

    # Unpack values from cfg.train.scheduler_params
    scheduler_type = cfg.train.scheduler
    decay_factor = cfg.train.scheduler_params.decay_factor
    decay_steps = cfg.train.scheduler_params.decay_steps
    patience = cfg.train.scheduler_params.patience
    warmup_epochs = cfg.train.scheduler_params.warmup_epochs
    warmup = warmup_epochs != -1

    if scheduler_type == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=decay_steps,
            gamma=1.0 / decay_factor,
        )
    elif scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=1.0 / decay_factor,
            patience=patience,
            verbose=True,
            # threshold_mode="rel",
            # min_lr=2.5e-4,
        )
    elif scheduler_type == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=decay_factor,
            last_epoch=-1,
        )
    elif scheduler_type == "cosine":
        size_dataset = DATASET_SIZES[cfg.dataset]

        if warmup:
            # If warmup is used, then we need to substract this from T_max.
            T_max = (cfg.train.epochs - warmup_epochs) * math.ceil(
                size_dataset / float(cfg.train.batch_size)
            )  # - warmup epochs
        else:
            T_max = cfg.train.epochs * math.ceil(
                size_dataset / float(cfg.train.batch_size)
            )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6,
        )
    else:
        lr_scheduler = None
        print(
            f"WARNING! No scheduler will be used. cfg.train.scheduler = {scheduler_type}"
        )

    if warmup and lr_scheduler is not None:
        size_dataset = DATASET_SIZES[cfg.dataset]

        lr_scheduler = ckconv.nn.LinearWarmUp_LRScheduler(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warmup_iterations=warmup_epochs
            * math.ceil(size_dataset / float(cfg.train.batch_size)),
        )

    return lr_scheduler
