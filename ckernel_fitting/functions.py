import numpy as np
import math
from PIL import Image
import skimage
import copy

import torch
import torchvision
from torchvision import transforms

import imageio


def get_function_to_fit(config):
    # form lin_space
    tensors = tuple(2 * [torch.linspace(config.min, config.max, config.no_samples)])
    x = torch.stack(torch.meshgrid(*tensors), dim=0)
    # select function
    function = {
        "Gaussian": _gaussian,
        "Constant": _constant,
        "Sawtooth": _sawtooth,
        "SineChirp": _sinus_chirp,
        "Random": _random,
        "CameraMan": _cameraman,
        "AlexNet": _alexnet,
        "CIFAR": _cifar,
        "Kodak": _kodak,
        "Sinus": _sinus,
        "Sine": _sine,
        "Gabor": _gabor,
    }[config.function]
    # apply
    sampled_function = function(config, x)
    return sampled_function


def _gaussian(config, x):
    # params
    mean = torch.tensor([0.0, 0.0]).view(2, 1, 1)
    distances = ((x - mean) ** 2).sum(dim=0, keepdim=True).sqrt()

    sigma = config.gauss_sigma
    # apply function
    f = (
        1
        / (sigma * math.sqrt(2.0 * math.pi))
        * torch.exp(-1 / 2.0 * (distances / sigma) ** 2)
    )
    f = 1 / float(torch.max(f)) * f

    # return
    return f


def _constant(config, x):
    x_shape = x.shape
    # apply function
    f = torch.ones_like(x[0, :, :])
    f[x_shape[-2] // 2 :, x_shape[-1] // 2 :] = 0.0
    f[: x_shape[-2] // 2, : x_shape[-1] // 2] = 0.0
    # return
    return f


def _sawtooth(config, x):
    # apply function
    f = torch.ones_like(x[0])
    f[::2, ::2] = 0.0

    # return
    return f


def _sinus(config, x):
    distances = (x ** 2).sum(dim=0, keepdims=True).sqrt()
    f = torch.sin(config.freq * distances ** 2)
    return f


def _sine(config, x):
    f = torch.sin(config.freq * x[0]) * torch.sin(config.freq * x[1])
    f = f.unsqueeze(0)
    return f


def _gabor(config, x):
    f = torch.sin(config.freq * x[0]) * torch.sin(config.freq * x[1])
    f = f.unsqueeze(0)
    g = _gaussian(config, x)
    return f * g


def _sinus_chirp(config, x):
    distances = (x ** 2).sum(dim=0).sqrt()
    # apply function
    f = torch.sin(distances ** 2)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _random(config, x):
    # apply function
    # f = torch.rand(config.no_images, *x[0].shape)
    with open("random_kernel.npy", "rb") as fd:
        np_f = np.load(fd)
    f = torch.tensor(np_f, dtype=torch.float32)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _cameraman(config, x):
    img = Image.fromarray(skimage.data.camera())
    transform = transforms.Compose(
        [
            transforms.Resize(x.shape[-1]),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        ]
    )
    img = transform(img)
    img = img.view(1, img.shape[-2], img.shape[-1])
    return img


def _alexnet(config, x):
    alexnet = torchvision.models.alexnet(pretrained=True)
    f = alexnet.features[0].weight.clone()  # The learned features at the first lyr.
    # Resize
    print(f.shape)
    f = torch.nn.Upsample(scale_factor=config.no_samples / 11.0, mode="bilinear")(f)
    f = f.view(-1, *f.shape[2:])
    print(f.shape)
    # return
    return f[0].unsqueeze(0)


def _cifar(config, x):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=transform, download=False
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.no_images,
        shuffle=False,
        num_workers=1,
    )

    # Gather first config.no_samples images from the dataset
    f = None
    for images, _ in loader:
        f = images.clone()[:, 0, :, :].unsqueeze(1)  # Take first channel of each image.
        # Resize
        f = torch.nn.Upsample(scale_factor=config.no_samples / 32.0, mode="bilinear")(f)
        print(f.shape)
        break
    f = f.view(-1, *f.shape[2:])

    # return
    return f


def _kodak(config, x):
    i = config.image_idx
    f = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    f = transforms.ToTensor()(f).float()

    return f
