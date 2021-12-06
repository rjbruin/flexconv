# torch
import hydra.utils
import torch

# built-in
import copy
import os
import datetime
import time
import numpy as np
import math

# logging
import wandb

# project
import probspec_routines as ps_routines
from tester import test
import ckconv
from torchmetrics import Accuracy
import antialiasing
from optim import construct_optimizer, construct_scheduler, CLASSES_DATASET

# typing
from typing import Dict
from omegaconf import OmegaConf


def save_to_wandb(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    cfg: OmegaConf,
    name: str = None,
    epoch: int = None,
):
    filename = f"{name}.pt"
    if epoch is not None:
        filename = "checkpoint.pt"
    path = os.path.join(wandb.run.dir, filename)

    torch.save(
        {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    # Call wandb to save the object, syncing it directly
    wandb.save(path)


def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    cfg: OmegaConf,
    epoch_start: int = 0,
):

    criterion = {
        "AddProblem": torch.nn.functional.mse_loss,
        "CopyMemory": torch.nn.CrossEntropyLoss,
        "MNIST": torch.nn.CrossEntropyLoss,
        "sMNIST": torch.nn.CrossEntropyLoss,
        "CIFAR10": torch.nn.CrossEntropyLoss,
        "sCIFAR10": torch.nn.CrossEntropyLoss,
        "CIFAR100": torch.nn.CrossEntropyLoss,
        "STL10": torch.nn.CrossEntropyLoss,
        "Cityscapes": torch.nn.CrossEntropyLoss,
        "VOC": torch.nn.CrossEntropyLoss,
        "Imagenet": torch.nn.CrossEntropyLoss,
        "Imagenet64": torch.nn.CrossEntropyLoss,
        "Imagenet32": torch.nn.CrossEntropyLoss,
        "Imagenet16": torch.nn.CrossEntropyLoss,
        "Imagenet8": torch.nn.CrossEntropyLoss,
        "SpeechCommands": torch.nn.CrossEntropyLoss,
        "CharTrajectories": torch.nn.CrossEntropyLoss,
    }[cfg.dataset]

    train_function = {
        "AddProblem": ps_routines.add_problem_train,
        "CopyMemory": ps_routines.copy_problem_train,
        "MNIST": classification_train,
        "sMNIST": classification_train,
        "CIFAR10": classification_train,
        "sCIFAR10": classification_train,
        "CIFAR100": classification_train,
        "Imagenet": classification_train,
        "Imagenet64": classification_train,
        "Imagenet32": classification_train,
        "Imagenet16": classification_train,
        "Imagenet8": classification_train,
        "STL10": classification_train,
        "SpeechCommands": classification_train,
        "CharTrajectories": classification_train,
    }[cfg.dataset]

    # Define optimizer and scheduler
    optimizer = construct_optimizer(model, cfg)
    lr_scheduler = construct_scheduler(optimizer, cfg)

    # train network
    _ = train_function(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        cfg=cfg,
        epoch_start=epoch_start,
    )

    save_to_wandb(model, optimizer, lr_scheduler, cfg, name="final_model")

    return model, optimizer, lr_scheduler


def classification_train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
    epoch_start: int = 0,
):
    # DEBUG
    # torch.autograd.set_detect_anomaly(True)

    weight_regularizer = ckconv.nn.LnLoss(
        weight_loss=cfg.train.weight_decay, norm_type=2
    )
    limit_regularizer = ckconv.nn.LimitLnLoss(
        weight_loss=cfg.train.mask_l2_norm, norm_type=2
    )
    # norm_regularizer = ckconv.nn.regularizers.MagnitudeRegularization(
    #     weight_loss=cfg.magnitude_reg, norm_type=2
    # ) # TODO: This necessary?

    # Permuter for psMNIST
    if cfg.dataset == "sMNIST" and cfg.dataset_params.permuted:
        permutation = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
        # Save in the config
        # cfg.dataset_params.permutation = permutation

    # Noise for noise-padded sCIFAR10
    if cfg.dataset == "sCIFAR10" and cfg.dataset_params.noise_padded:
        rands = torch.randn(1, 1000 - 32, 96)

    # Training parameters
    epochs = cfg.train.epochs
    # Testcases: override epochs
    if cfg.testcase.load or cfg.testcase.save:
        epochs = cfg.testcase.epochs
    device = cfg.device

    criterion = criterion().to(device)

    # Log limits, before training
    log_limits(model, step=0)

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_top5 = 0.0
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = 100

    if cfg.testcase.save or cfg.testcase.load:
        testcase_losses = []

    # iterate over epochs
    for epoch in range(epoch_start, epochs + epoch_start):
        print("Epoch {}/{}".format(epoch + 1, epochs + epoch_start))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            phase_start_time = time.time()

            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            running_gabor_reg = 0.0
            total = 0

            if phase == "validation" and cfg.train.report_top5_acc:

                top5 = Accuracy(
                    num_classes=CLASSES_DATASET[cfg.dataset],
                    top_k=5,
                    compute_on_step=False,
                )

            # iterate over data
            for data in dataloaders[phase]:
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

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    inputs = torch.dropout(inputs, cfg.net.dropout_in, train)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # Regularization:
                    if cfg.train.weight_decay > 0.0:
                        loss = loss + weight_regularizer(model)
                    if cfg.train.mask_l2_norm > 0.0:
                        loss = loss + limit_regularizer(model)
                    # if cfg.magnitude_reg > 0.0:
                    #     loss = loss + norm_regularizer(model)
                    if cfg.kernel.regularize:
                        gabor_reg = antialiasing.regularize_gabornet(
                            model,
                            cfg.kernel.regularize_params.res,
                            cfg.kernel.regularize_params.factor,
                            cfg.kernel.regularize_params.target,
                            cfg.kernel.regularize_params.fn,
                            cfg.kernel.regularize_params.method,
                            gauss_stddevs=cfg.kernel.regularize_params.gauss_stddevs,
                        )
                        loss += gabor_reg
                        running_gabor_reg += gabor_reg

                        # DEBUG
                        # modules = antialiasing.get_flexconv_modules(model)
                        # for t in ['sines', 'gausses', 'gabor']:
                        #     freqs = []
                        #     for module in modules:
                        #         freqs.append(antialiasing.gabor_layer_frequencies(module, t, config.regularize_gabornet_method))
                        #     freqs = torch.stack(freqs)
                        #     print(f"{t} frequencies: {freqs[0]}")
                        # print(f"Lambda: {config.regularize_gabornet_lambda}")
                        # print(f"Resolution: {config.regularize_gabornet_res}")
                        # print(f"Total regularization term (incl. lambda): {gabor_reg:.8f}")

                    if cfg.testcase.save or cfg.testcase.load:
                        testcase_losses.append(loss.item())

                    # Backward pass:
                    if phase == "train":
                        loss.backward()
                        if cfg.train.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.train.grad_clip
                            )
                        optimizer.step()

                        # update the lr_scheduler
                        if isinstance(
                            lr_scheduler,
                            (
                                torch.optim.lr_scheduler.CosineAnnealingLR,
                                ckconv.nn.LinearWarmUp_LRScheduler,
                            ),
                        ):
                            lr_scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)
                if phase == "validation" and cfg.train.report_top5_acc:
                    pred_sm = torch.nn.functional.softmax(outputs, dim=1)
                    # torchmetrics.Accuracy requires everything to be on CPU
                    top5(pred_sm.to("cpu"), labels.to("cpu"))

                if total >= cfg.testcase.batches:
                    break

            # Log GaborNet frequencies
            if cfg.kernel.regularize and phase == "train":
                stats = antialiasing.get_gabornet_summaries(
                    model,
                    cfg.kernel.regularize_params.target,
                    cfg.kernel.regularize_params.method,
                )
                wandb.log(stats, step=epoch + 1)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            epoch_gabor_reg = running_gabor_reg / total
            if phase == "validation" and cfg.train.report_top5_acc:
                epoch_top5 = top5.compute()
                print(
                    "{} Loss: {:.4f} Acc: {:.4f} Top-5: {:.4f}".format(
                        phase, epoch_loss, epoch_acc, epoch_top5
                    )
                )
            else:
                epoch_top5 = 0.0
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )
            print(f"GaborNet regularization: {epoch_gabor_reg:.8f}")
            print(datetime.datetime.now())
            phase_end_time = time.time()
            phase_time = phase_end_time - phase_start_time

            # log statistics of the epoch
            wandb.log(
                {
                    "accuracy" + "_" + phase: epoch_acc,
                    "accuracy_top5" + "_" + phase: epoch_top5,
                    "loss" + "_" + phase: epoch_loss,
                    "gabor_reg" + "_" + phase: epoch_gabor_reg,
                    phase + "_time": phase_time,
                },
                step=epoch + 1,
            )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_top5 = epoch_top5
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_to_wandb(model, optimizer, lr_scheduler, cfg, epoch=epoch + 1)

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results
                    if cfg.dataset in ["SpeechCommands", "CharTrajectories"]:
                        test_acc, test_top5 = test(model, dataloaders["test"], cfg)
                    else:
                        test_acc = best_acc
                        test_top5 = best_top5
                    wandb.run.summary["best_test_accuracy"] = test_acc
                    wandb.run.summary["best_test_top5"] = test_top5
                    wandb.log(
                        {"accuracy_test": test_acc, "accuracy_top5_test": test_top5},
                        step=epoch + 1,
                    )

                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0

            elif phase == "validation" and epoch_acc < best_acc:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            # Log limits
            log_limits(model, epoch + 1)

            # Update scheduler
            if (
                isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                and phase == "validation"
            ):
                lr_scheduler.step(epoch_acc)

        # Update scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR) or isinstance(
            lr_scheduler, torch.optim.lr_scheduler.ExponentialLR
        ):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy."
            )
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    if cfg.train.report_top5_acc:
        print("Best Val Top-5: {:.4f}".format(best_top5))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Print learned limits
    _print_learned_limits(model)

    # Testcases: load/save losses for comparison
    if cfg.testcase.save:
        testcase_losses = np.array(testcase_losses)
        with open(hydra.utils.to_absolute_path(cfg.testcase.path), 'wb') as f:
            np.save(f, testcase_losses, allow_pickle=True)
    if cfg.testcase.load:
        testcase_losses = np.array(testcase_losses)
        with open(hydra.utils.to_absolute_path(cfg.testcase.path), 'rb') as f:
            target_losses = np.load(f, allow_pickle=True)
        if np.allclose(testcase_losses, target_losses):
            print("Testcase passed!")
        else:
            diff = np.sum(testcase_losses - target_losses)
            raise AssertionError(f"Testcase failed: diff = {diff:.8f}")

    # Return model
    return model


def log_limits(model, step):
    log = {}
    limitss = get_limits(model)
    for i, limits in enumerate(limitss):
        log.update({f"limit_{i}_{k}": v for (k, v) in limits.items()})
    wandb.log(log, step=step)


def get_limits(model):
    limitss = []
    for m in model.modules():
        if isinstance(m, ckconv.nn.FlexConv):
            limits = m.mask_params.detach().cpu()
            # top, bottom, left, right
            if m.kernel_dim_linear == 1:
                (mean, std) = limits.squeeze()
                limitss.append(
                    {
                        "mean": mean,
                        "std": std,
                    }
                )
            elif m.kernel_dim_linear == 2:
                (mean_y, std_y), (mean_x, std_x) = limits
                limitss.append(
                    {
                        "mean_y": mean_y,
                        "std_y": std_y,
                        "mean_x": mean_x,
                        "std_x": std_x,
                    }
                )
    return limitss


def _print_learned_limits(model):
    limits_final = []
    print(50 * "-")
    print("Learned limits:")
    for m in model.modules():
        if isinstance(m, ckconv.nn.FlexConv):
            limits = m.mask_params.detach().cpu()
            limits_final.append(limits)
            print(limits)
    print(50 * "-")
