import torch
import torch.nn.functional as F
import numpy as np
import timer

import wandb
from torchmetrics import IoU, Accuracy

# project
import probspec_routines as ps_routines
from optim import CLASSES_DATASET

# typing
from omegaconf import OmegaConf


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cfg: OmegaConf,
    name: str = None,
    **kwargs,
):

    test_function = {
        "AddProblem": ps_routines.add_problem_test,
        "CopyMemory": ps_routines.copy_problem_test,
        "SpeechCommands": classification_test,
        "CharTrajectories": classification_test,
        "MNIST": classification_test,
        "sMNIST": classification_test,
        "CIFAR10": classification_test,
        "sCIFAR10": classification_test,
        "CIFAR100": classification_test,
        "Imagenet": classification_test,
        "Imagenet64": classification_test,
        "Imagenet32": classification_test,
        "Imagenet16": classification_test,
        "Imagenet8": classification_test,
        "STL": classification_test,
    }[cfg.dataset]

    return test_function(model, test_loader, cfg, name=name)


def classification_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cfg: OmegaConf,
    log: bool = False,
    epoch: int = None,
    name: str = None,
):
    # send model to device
    device = cfg.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0

    # Permuter for psMNIST
    if cfg.dataset == "sMNIST" and cfg.dataset_params.permuted:
        # Check if the config file has the key, otherwise create the permutation. #TODO: Not supported by hydra
        # if "permutation" in cfg.dataset_params:
        #     permutation = cfg.dataset_params.permutation
        # else:
        permutation = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

    # Noise for noise-padded sCIFAR10
    if cfg.dataset == "sCIFAR10" and cfg.dataset_params.noise_padded:
        rands = torch.randn(1, 1000 - 32, 96)

    if cfg.train.report_top5_acc:
        top5 = Accuracy(
            num_classes=CLASSES_DATASET[cfg.dataset],
            top_k=5,
            compute_on_step=False,
        )

    with torch.no_grad():
        # Iterate through data
        for data in test_loader:
            # DALI has a different dataloader output format
            if cfg.dataset == "Imagenet":
                data = (data[0]["data"], data[0]["label"].squeeze(1))
            inputs, labels = data

            # Add padding if noise_padding
            if cfg.dataset_params.noise_padded and cfg.dataset == "sCIFAR10":
                inputs = torch.cat(
                    (
                        inputs.permute(0, 2, 1, 3).reshape(inputs.shape[0], 32, 96),
                        rands.repeat(inputs.shape[0], 1, 1),
                    ),
                    dim=1,
                ).permute(0, 2, 1)
            else:
                # Make sequential if sMNIST or sCIFAR10
                if cfg.dataset in ["sMNIST", "sCIFAR10"]:
                    _, in_channels, x, y = inputs.shape
                    inputs = inputs.view(-1, in_channels, x * y)
                # Permute if psMNIST
                if cfg.dataset_params.permuted and cfg.dataset == "sMNIST":
                    inputs = inputs[:, :, permutation]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if cfg.train.report_top5_acc:
                pred_sm = torch.nn.functional.softmax(outputs, dim=1)
                # torchmetrics.Accuracy requires everything to be on CPU
                top5(pred_sm.to("cpu"), labels.to("cpu"))

    # Print results
    test_acc = correct / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )

    test_top5 = 0.0
    if cfg.train.report_top5_acc:
        test_top5 = top5.compute()
        print(
            "Top-5 accuracy of the network on the {} test samples: {}".format(
                total, (100 * test_top5)
            )
        )

    return test_acc, test_top5
