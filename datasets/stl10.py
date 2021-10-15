from torchvision import datasets, transforms
from ckconv.utils.cutout import Cutout

from hydra import utils
import os


class STL10(datasets.STL10):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
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
        if augment == "resnet":
            transform = get_augmentations()
        elif augment == "None":
            transform = [
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
                ),
            ]
        else:
            raise NotImplementedError(f"augment = {augment}")

        # Resize according to config.resize
        if "resize" in kwargs and kwargs["resize"] != "":
            try:
                img_size = list(map(int, kwargs["resize"].split(",")))
            except Exception:
                raise ValueError(
                    f"config.resize \"{kwargs['resize']}\" is in "
                    f'wrong format. Should be "(h, w)".'
                )
            transform.append(transforms.Resize(img_size))

        transform = transforms.Compose(transform)

        super().__init__(root=root, split=partition, transform=transform, download=True)


def get_augmentations():
    augmentations = [
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        Cutout(1, 32),
    ]
    return augmentations
