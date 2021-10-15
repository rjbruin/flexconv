import torch
from datasets import (
    # 1D
    AdditionProblem,
    CopyMemory,
    SpeechCommands,
    CharTrajectories,
    # 2D classification
    MNIST,
    CIFAR10,
    CIFAR100,
    STL10,
    ImagenetDownsampled,
)

# typing
from omegaconf import OmegaConf
from typing import Dict, Tuple
from torch.utils.data import DataLoader

DATASET_RESOLUTIONS = {
    "CIFAR10": 32,
    "CIFAR100": 32,
}


def dataset_constructor(
    cfg: OmegaConf,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple (training_set, validation_set, test_set)
    """
    dataset = {
        "AddProblem": AdditionProblem,
        "CopyMemory": CopyMemory,
        "MNIST": MNIST,
        "sMNIST": MNIST,
        "CIFAR10": CIFAR10,
        "sCIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
        "Imagenet64": ImagenetDownsampled,
        "Imagenet32": ImagenetDownsampled,
        "Imagenet16": ImagenetDownsampled,
        "Imagenet8": ImagenetDownsampled,
        "Imagenet-k": ImagenetDownsampled,
        "STL10": STL10,
        "SpeechCommands": SpeechCommands,
        "CharTrajectories": CharTrajectories,
    }[cfg.dataset]

    test_partition = "test"

    # Custom settings for some datasets, passed by keyword args
    kwargs = {}
    if cfg.dataset in ["Imagenet64", "Imagenet32", "Imagenet16", "Imagenet8"]:
        kwargs["root"] = cfg.dataset_params.root
        kwargs["use_ram"] = cfg.dataset_params.in_ram
        kwargs["use_cache"] = cfg.dataset_params.from_cache
        kwargs["img_size"] = int(cfg.dataset.split("Imagenet")[1])
        test_partition = "val"
    elif cfg.dataset == "CIFAR10":
        kwargs["blur"] = cfg.cross_res.blur

    training_set = dataset(
        partition="train",
        seq_length=cfg.dataset_params.seq_length,
        memory_size=cfg.dataset_params.memory_size,
        mfcc=cfg.dataset_params.mfcc,
        dropped_rate=cfg.dataset_params.drop_rate,
        augment=cfg.train.augment,
        resize=cfg.cross_res.resize,
        resize_interpolation=cfg.cross_res.interpolation,
        resize_blur=cfg.cross_res.resize_blur,
        resize_blur_sigma=cfg.cross_res.resize_blur_sigma,
        **kwargs,
    )
    test_set = dataset(
        partition=test_partition,
        seq_length=cfg.dataset_params.seq_length,
        memory_size=cfg.dataset_params.memory_size,
        mfcc=cfg.dataset_params.mfcc,
        dropped_rate=cfg.dataset_params.drop_rate,
        augment="None",
        resize=cfg.cross_res.resize,
        resize_interpolation=cfg.cross_res.interpolation,
        resize_blur=cfg.cross_res.resize_blur,
        resize_blur_sigma=cfg.cross_res.resize_blur_sigma,
        **kwargs,
    )
    if cfg.dataset in ["SpeechCommands", "CharTrajectories"]:
        validation_set = dataset(
            partition="val",
            seq_length=cfg.dataset_params.seq_length,
            memory_size=cfg.dataset_params.memory_size,
            mfcc=cfg.dataset_params.mfcc,
            dropped_rate=cfg.dataset_params.drop_rate,
            augment="None",
            resize=cfg.cross_res.resize,
            resize_interpolation=cfg.cross_res.interpolation,
            resize_blur=cfg.cross_res.resize_blur,
            resize_blur_sigma=cfg.cross_res.resize_blur_sigma,
            **kwargs,
        )
    else:
        validation_set = None
    return training_set, validation_set, test_set


def construct_dataloaders(
    cfg: OmegaConf,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: dict(train_loader, val_loader, test_loader)
    """
    training_set, validation_set, test_set = dataset_constructor(cfg)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_loader = test_loader

    dataloaders = {
        "train": training_loader,
        "validation": val_loader,
        "test": test_loader,
    }

    return dataloaders
