from torchvision import datasets, transforms
import torch

from hydra import utils
import os


class MNIST(datasets.MNIST):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            root = os.path.join(root, "data")

        if "resize_blur" in kwargs and kwargs["resize_blur"]:
            raise NotImplementedError()

        augment = kwargs["augment"]
        if augment == "standard":
            transform = get_augmentations()
        elif augment == "None":
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        else:
            raise NotImplementedError(f"augment = {augment}")

        transform = transforms.Compose(transform)

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super(MNIST, self).__init__(
            root=root, train=train, transform=transform, download=True
        )


def get_augmentations():
    """
    Following "A branching and merging convolutional network with homogeneous filter capsules"
    - Biearly et al., 2020 - https://arxiv.org/abs/2001.09136
    """
    augmentations = [
        transforms.RandomApply(
            [transforms.RandomRotation(30)], p=0.5
        ),  # Rotation by 30 degrees with probability 0.5
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(2 / 28.0, 2 / 28.0))], p=0.5
        ),  # Translation of 2 pixels with probability 0.5
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.75, 1))],
            p=0.5,
        ),  # Rescale with probability 0.5
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(
            p=0.5, scale=(4 / 28.0, 4 / 28.0), ratio=(1.0, 1.0), value=0, inplace=False
        ),  # Erase patches of 4 pixels with probability 0.5
    ]
    return augmentations
