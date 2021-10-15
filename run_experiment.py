# general
import os
import random
import sys
import copy
import time

# torch
import numpy as np
import torch

# project
from path_handler import model_path
from model_constructor import construct_model
from dataset_constructor import construct_dataloaders, get_imagenet_dali_tfrecords
import trainer
import tester
import ckconv

# Loggers and config
import wandb
import hydra
from omegaconf import OmegaConf
from hydra import utils
import torchinfo


def setup(cfg):
    # With -1 seed, we use a determined and stored random seed instead
    if cfg.seed == -1:
        cfg.seed = np.random.randint(0, 100)

    # Set the seed
    set_manual_seed(cfg.seed, cfg.deterministic)

    # Initialize wandb
    if not cfg.train or cfg.debug:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["HYDRA_FULL_ERROR"] = "1"

    wandb.init(
        project=cfg.wandb.project,
        config=ckconv.utils.flatten_configdict(cfg),
        entity=cfg.wandb.entity,
        save_code=True,
        dir=cfg.wandb.dir,
    )


def model_and_datasets(cfg):
    # Construct the model
    model = construct_model(cfg)
    # Send model to GPU if available, otherwise to CPU
    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())

    if cfg.device == "cuda" and torch.cuda.is_available():
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        # Set device and send model to device
        cfg.device = "cuda"
        model.to(cfg.device)

    # Construct dataloaders
    dataloaders = construct_dataloaders(cfg)
    if cfg.dataset == "Imagenet":
        # NOTE(rjbruin): ImageNet integration uses the NVIDIA DALI interface for
        # fast training, making it incompatible with the general API
        dataloaders = get_imagenet_dali_tfrecords(cfg)

    # # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200)

    # Create model directory and instantiate config.path
    # model_path(cfg) # TODO

    if cfg.pretrained:
        # Load model state dict
        missing, unexpected = model.module.load_state_dict(
            torch.load(cfg.pretrained_params.filepath, map_location=cfg.device)[
                "model"
            ],
            strict=cfg.pretrained_strict,
        )
        print("Loaded model.")
    elif cfg.pretrained_wandb:
        # Load model state dict from wandb
        weights_file = wandb.restore(
            cfg.pretrained_wandb_params.filename,
            run_path=cfg.pretrained_wandb_params.run_path,
        )
        missing, unexpected = model.module.load_state_dict(
            torch.load(weights_file.name, map_location=cfg.device)["model"],
            strict=cfg.pretrained_strict,
        )
        print("Loaded model from W&B.")

    if cfg.pretrained or cfg.pretrained_wandb:
        if len(missing) > 0:
            print("Missing keys:\n" + "\n".join(missing))
        if len(unexpected) > 0:
            print("Unexpected keys:\n" + "\n".join(unexpected))
        # Clear train lengths
        for m in model.modules():
            if isinstance(m, ckconv.nn.CKConv):
                m.train_length[0] = 0

    if len(cfg.summary) > 1:
        torchinfo.summary(model, tuple(cfg.summary), depth=cfg.summary_depth)

    return model, dataloaders


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)
    # Print input args
    print(f"Input arguments \n {OmegaConf.to_yaml(cfg)}")

    setup(cfg)
    model, dataloaders = model_and_datasets(cfg)

    if cfg.test.before_train:
        tester.test(model, dataloaders["test"], cfg, log=True, epoch=0)

    # Train the model
    if cfg.train.do:
        # Print arguments (Sanity check)
        print(f"Modified arguments: \n {OmegaConf.to_yaml(cfg)}")

        # Train the model
        trainer.train(model, dataloaders, cfg)

    # Select test function
    tester.test(model, dataloaders["test"], cfg)


def set_manual_seed(
    seed: int,
    deterministic: bool,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
