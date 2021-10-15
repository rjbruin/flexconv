import math
import os
from hydra import utils
from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            # DEBUG
            # root = "../"
            root = os.path.join(root, "data")

        transform = []
        resize_to = None
        resize_blur = False

        # Resize according to config.resize
        if "resize" in kwargs and kwargs["resize"] != "":
            try:
                img_size = list(map(int, kwargs["resize"].split(",")))
            except Exception:
                raise ValueError(
                    f"config.resize \"{kwargs['resize']}\" is in "
                    f"wrong format. Should be `h,w`."
                )
            resize_to = img_size[0]
            resize_blur = kwargs["resize_blur"] if "resize_blur" in kwargs else False
            resize_blur_sigma = (
                kwargs["resize_blur_sigma"] if "resize_blur_sigma" in kwargs else 1.0
            )

        # Blur anyway, regardless of resize
        if "blur" in kwargs and kwargs["blur"]:
            resize_blur = True
            resize_blur_sigma = (
                kwargs["resize_blur_sigma"] if "resize_blur_sigma" in kwargs else 1.0
            )

        # Blur before resize, regardless of partition
        if resize_blur and resize_blur_sigma != 0.0:
            # Half kernel size = 3 x sigma, rounded up
            kernel_size = math.ceil(resize_blur_sigma * 3.0) * 2 + 1
            transform.append(
                transforms.GaussianBlur(kernel_size, sigma=resize_blur_sigma)
            )

        augment = kwargs["augment"]
        if augment == "resnet":
            transform.extend(
                augmentations_resnet(
                    resize_to=resize_to, interpolation=kwargs["resize_interpolation"]
                )
            )
        elif augment == "None":
            if resize_to is not None:
                interpolation = {
                    "bilinear": transforms.InterpolationMode.BILINEAR,
                    "nearest": transforms.InterpolationMode.NEAREST,
                }[kwargs["resize_interpolation"]]
                transform.append(
                    transforms.Resize(resize_to, interpolation=interpolation)
                )
            transform.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
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

        super().__init__(root=root, train=train, transform=transform, download=True)


# def get_augmentations():
#     """
#     Following "A branching and merging convolutional network with homogeneous filter capsules"
#     - Biearly et al., 2020 - https://arxiv.org/abs/2001.09136
#     """
#     augmentations = [
#         transforms.RandomApply(
#             [transforms.RandomRotation(30)], p=0.5
#         ),  # Rotation by 30 degrees with probability 0.5
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.75, 1))],
#             p=0.5,
#         ),  # Rescale with probability 0.5
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         transforms.RandomErasing(
#             p=0.5, scale=(4 / 32.0, 4 / 32.0), ratio=(1.0, 1.0), value=0, inplace=False
#         ),  # Erase patches of 4 pixels with probability 0.5
#     ]
#     return augmentations


def augmentations_resnet(resize_to=None, crop_size=None, interpolation="bilinear"):
    """
    Following "A branching and merging convolutional network with homogeneous filter capsules"
    - Biearly et al., 2020 - https://arxiv.org/abs/2001.09136
    """
    if crop_size is None:
        crop_size = 32
    pad_size = crop_size // 8

    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(crop_size, pad_size),
    ]
    if resize_to is not None:
        if interpolation == "bilinear":
            interpolation = transforms.InterpolationMode.BILINEAR
        elif interpolation == "nearest":
            interpolation = transforms.InterpolationMode.NEAREST
        else:
            raise NotImplementedError(f"resize_interpolation={interpolation}")
        augmentations.append(transforms.Resize(resize_to, interpolation=interpolation))

    augmentations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return augmentations
