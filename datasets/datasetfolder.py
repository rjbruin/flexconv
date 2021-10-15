"""
Extended versions of torchvision's dataset definitions.
"""
import os
import torch
import torchvision
import numpy as np
from PIL import Image
import tqdm
import sys

from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS


class RAMableDatasetFolder(torchvision.datasets.VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    This data loader loads all samples into memory on initialization, instead
    of loading samples on iteration over the dataset.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples.
        images (list): List of pre-loaded PIL images.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(
        self,
        root,
        loader,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        use_ram=False,
        use_cache=False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.loader = loader
        self.extensions = extensions
        self.use_ram = use_ram
        self.use_cache = use_cache

        if self.use_cache:
            self.load()
        else:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                raise (
                    RuntimeError(
                        "Found 0 files in subfolders of: " + self.root + "\n"
                        "Supported extensions are: " + ",".join(extensions)
                    )
                )

            self.classes = classes
            self.class_to_idx = class_to_idx
            self.samples = samples
            self.targets = [s[1] for s in samples]

            # Load all samples into RAM at initialization time
            if self.use_ram:
                self.load_all()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def load_all(self):
        """
        Preload all images without transformations into memory.
        Transformations are to be applied when accessing the dataset.

        Throws:
            MemoryError: when the dataset cannot fit in memory.
        """
        # Discover size of images, so that we can allocate memory efficiently
        # NOTE(rjbruin): assumes that all images have the same size!
        n_images = len(self.samples)
        sample_image = np.array(self.loader(self.samples[0][0]))
        height = sample_image.shape[0]
        width = sample_image.shape[1]
        n_channels = sample_image.shape[2]

        self.images = np.zeros((n_images, height, width, n_channels), dtype="uint8")
        try:
            for i, sample in tqdm.tqdm(
                enumerate(self.samples),
                desc="Pre-loading dataset",
                total=len(self.samples),
            ):
                path, _ = sample
                self.images[i] = np.array(self.loader(path))
        except MemoryError:
            raise MemoryError(
                "Dataset cannot fit in memory! Please run "
                "without --ram-dataset or use more memory."
            )

    @property
    def preloaded(self):
        return self.use_ram and len(self.images) > 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.preloaded:
            sample = self.images[index]
            _, target = self.samples[index]
            # NOTE(rjbruin): for some reason, the cache has str's for the targets
            target = int(target)
        else:
            path, target = self.samples[index]
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def save(self):
        images = self.images

        # Convert images to NumPy if necessary
        if isinstance(images[0], Image.Image):
            for i in tqdm.tqdm(
                range(len(images)), desc="Converting images to NumPy for saving"
            ):
                images[i] = np.array(images[i])
            images = np.array(images)
        else:
            print(f"Images already in NumPy!")

        print(f"Saving dataset...")
        with open(os.path.join(self.root, "images.npy"), "wb") as f:
            np.save(f, images)
        with open(os.path.join(self.root, "samples.npy"), "wb") as f:
            np.save(f, self.samples)
        with open(os.path.join(self.root, "classes.npy"), "wb") as f:
            np.save(f, self.classes)
        with open(os.path.join(self.root, "class_to_idx.npy"), "wb") as f:
            np.save(f, self.class_to_idx)
        with open(os.path.join(self.root, "targets.npy"), "wb") as f:
            np.save(f, self.targets)
        print(f"Dataset saved.")

    def load(self):
        print("Loading dataset from cache...")
        with open(os.path.join(self.root, "images.npy"), "rb") as f:
            self.images = np.load(f, allow_pickle=True)
        with open(os.path.join(self.root, "samples.npy"), "rb") as f:
            self.samples = np.load(f, allow_pickle=True)
        with open(os.path.join(self.root, "classes.npy"), "rb") as f:
            self.classes = np.load(f, allow_pickle=True)
        with open(os.path.join(self.root, "class_to_idx.npy"), "rb") as f:
            self.class_to_idx = np.load(f, allow_pickle=True)
        with open(os.path.join(self.root, "targets.npy"), "rb") as f:
            self.targets = np.load(f, allow_pickle=True)

        # NOTE(rjbruin): removed converting images to PIL, as nothing in the
        # further pipeline requires PIL Images instead of Numpy arrays (since
        # most torchvision transforms work interchangeably on PIL Images and
        # Numpy arrays), and the conversion doubles the required memory because
        # it has to store the Numpy and PIL representations of the dataset
        # simultaneously.

        # Convert images to PIL
        # new_images = [] for i in tqdm.tqdm(range(len(self.images)),
        # desc='Converting cache to PIL'):
        # new_images.append(Image.fromarray(self.images[i])) self.images =
        # new_images

        print("Dataset loaded.")


class RAMableImageFolder(RAMableDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    This data loader loads all samples into memory on initialization, instead
    of loading samples on iteration over the dataset.

    Args:
        root (string): Root directory path.
        transform (callabtorchvision.datasets.VisionDatasete, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
        loaded_imgs (list): List of pre-loaded PIL images.
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        use_ram=False,
        use_cache=False,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            use_ram=use_ram,
            use_cache=use_cache,
        )
