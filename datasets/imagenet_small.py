import os
import torchvision

from datasets.datasetfolder import RAMableImageFolder
from datasets.cifar10 import augmentations_resnet


class ImagenetDownsampled(RAMableImageFolder):
    def __init__(
        self,
        partition,
        root=None,
        augment="None",
        use_ram=False,
        use_cache=False,
        img_size=None,
        **kwargs,
    ):
        if root is None or root == "":
            raise ValueError("Missing dataset root.")

        root = os.path.join(root, partition)

        if "resize" in kwargs and kwargs["resize"] != "":
            raise NotImplementedError("resize for ImagenetDownsampled")
        if "resize_blur" in kwargs and kwargs["resize_blur"]:
            raise NotImplementedError()

        if augment == "resnet":
            transform = augmentations_resnet(crop_size=img_size)
            if use_ram:
                transform.insert(0, torchvision.transforms.ToPILImage())
        elif augment == "None":
            transform = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        else:
            raise NotImplementedError(f"augment = {augment}")
        transform = torchvision.transforms.Compose(transform)

        super().__init__(
            root,
            transform=transform,
            use_ram=use_ram,
            use_cache=use_cache,
        )
