"""
In this experiments, we analyze the fitting properties of the CKConv parameterization.
In particular we ask the following question:

 Which kind of functions are we able to fit via continuous kernels?
"""
# Append .. to path
import os, sys

ckconv_source = os.path.join(os.getcwd(), "..")
if ckconv_source not in sys.path:
    sys.path.append(ckconv_source)

# general
import os
import wandb
import ml_collections
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import csv

# torch
import numpy as np
import torch

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# project
import ckconv.nn
from ckernel_fitting.functions import get_function_to_fit
from ckconv.utils import grids

from srf_fitting.nn import Srf_layer_shared


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py")


def main(_):
    freqs = torch.logspace(0, 6, base=2.0, steps=4)
    # orders = torch.linspace(1, 5, steps=5)
    # orders = torch.linspace(1, 9, steps=9)
    # freqs = torch.tensor([64])
    # orders = torch.tensor([0])

    # results = {
    #     'mse': torch.zeros((len(freqs), len(orders)), dtype=torch.float32),
    #     'psnr': torch.zeros((len(freqs), len(orders)), dtype=torch.float32),
    # }
    results = {
        "mse": torch.zeros((len(freqs)), dtype=torch.float32),
        "psnr": torch.zeros((len(freqs)), dtype=torch.float32),
    }
    # results = {
    #     'mse': torch.zeros((len(orders)), dtype=torch.float32),
    #     'psnr': torch.zeros((len(orders)), dtype=torch.float32),
    # }
    # results = {
    #     'mse': torch.zeros((1), dtype=torch.float32),
    #     'psnr': torch.zeros((1), dtype=torch.float32),
    # }

    # i_f = 0
    # freq = 0

    for i_f, freq in enumerate(freqs):
        # for j_o, order in enumerate(orders):

        if "absl.logging" in sys.modules:
            import absl.logging

            absl.logging.set_verbosity("info")
            absl.logging.set_stderrthreshold("info")

        config = FLAGS.config
        print(config)

        config.freq = int(freq.item())
        # config.order = int(order.item())

        # Set the seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if config.debug:
            os.environ["WANDB_MODE"] = "dryrun"

        # initialize weight and bias
        wandb.init(
            project="kernel_approximation",
            config=copy.deepcopy(dict(config)),
            group=config.function,
            entity="vu_tudelft_team",
            tags=[config.function],
            save_code=True,
        )

        # Load the model: The model is always equal to a continuous kernel
        model = get_model(config)

        # get function to fit
        f = get_function_to_fit(config)
        if config.padding != 0:
            f = torch.nn.functional.pad(
                f, (config.padding, config.padding, config.padding, config.padding)
            )
        # plot function to fit
        if config.plot_target:
            plot_target(f)
        # plot_function_to_fit(f, config)

        # input to the model
        x = grids.rel_positions_grid(f.shape[1:]).unsqueeze(0)
        f = f.to(config.device).unsqueeze(0)

        # define optimizer
        # --------------
        all_parameters = set(model.parameters())
        omega_0s = []
        for m in model.modules():
            if isinstance(m, (ckconv.nn.MultipliedLinear2d, ckconv.nn.kernelnet.Sine)):
                omega_0s += list(
                    map(
                        lambda x: x[1],
                        list(
                            filter(lambda kv: "omega_0" in kv[0], m.named_parameters())
                        ),
                    )
                )
        omega_0s = set(omega_0s)
        other_params = all_parameters - omega_0s
        omega_0s = list(omega_0s)
        other_params = list(other_params)

        optimizer_class = getattr(torch.optim, config.optimizer)
        optimizer = optimizer_class(
            [
                {"params": other_params},
                {"params": omega_0s, "lr": 50 * config.lr},
            ],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.no_iterations,
            )
        elif config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=1.0 / config.sched_decay_factor,
                patience=config.sched_patience,
                verbose=True,
            )
        else:
            scheduler = None

        if config.pretrained:
            # Load model state dict
            model.module.load_state_dict(
                torch.load("saved/model.pt", map_location="cuda:0"), strict=False
            )

        # Send all input tensors to same device:
        # ------------------------------------
        # Set device
        config.device = (
            "cuda:0"
            if (config.device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        torch.backends.cudnn.benchmark = True

        model.to(device=config.device)
        f = f.to(device=config.device)
        x = x.to(device=config.device)

        if config.train:

            # Fit the kernel
            # --------------
            log_interval = config.log_interval
            # Define optimizer
            iter = 1
            total_loss = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy_train = 999

            for iterations in range(config.no_iterations):
                model.train()
                optimizer.zero_grad()

                output = model(x)
                loss = torch.nn.functional.mse_loss(output, f)

                loss.backward()
                optimizer.step()

                # scheduler
                if config.scheduler == "cosine":
                    scheduler.step()
                elif config.scheduler == "plateau":
                    scheduler.step(loss)

                iter += 1
                total_loss += loss.item()

                if iter % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    print(
                        "Iter: {:2d}/{:6d} \tLearning rate: {:.4f}\tLoss: {:.6f}".format(
                            iter, config.no_iterations, config.lr, cur_loss
                        )
                    )

                    print(f"PSNR: {ckconv.utils.psnr(f, torch.clamp(output, 0., 1.))}")
                    total_loss = 0

                    # log PSNR + MSE loss
                    wandb.log({"MSE": loss.item()}, step=iter)
                    wandb.log(
                        {
                            "PSNR": ckconv.utils.psnr(
                                f, torch.clamp(output, 0.0, 1.0)
                            ).item()
                        },
                        step=iter,
                    )
                    # Log lr
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=iter)

                if iter % config.model_save_interval == 0:
                    save_model_to_wandb(model, epoch=iter)

                    plot_gt = iter == config.model_save_interval
                    if config.plot_fit:
                        plot_fitted_kernel(
                            output.detach().cpu().squeeze(),
                            f.detach().cpu().squeeze(),
                            loss,
                            config,
                            plot_gt,
                            iter,
                        )

                if loss.item() < best_accuracy_train:
                    best_accuracy_train = loss.item()
                    wandb.run.summary["best_acc_train"] = best_accuracy_train
                    best_model_wts = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_model_wts)

            # Print learned w0s
            if config.learn_omega_0:
                print(50 * "-")
                print("Learned w0 values:")
                for m in model.modules():
                    if isinstance(m, ckconv.nn.MultipliedLinear1d) or isinstance(
                        m, ckconv.nn.MultipliedLinear2d
                    ):
                        print(m.omega_0.detach().cpu())
                print(50 * "-")

            # Save the model
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
            torch.save(model.state_dict(), "saved/model.pt")
            # --------------

        # Check the fitting
        # -----------------
        loss, psnr = evaluate(model, x, f, config)
        save_model_to_wandb(model, name="final_model")

        wandb.run.summary["finalMSE"] = loss.item()
        print("\nTest: loss: {}\n".format(loss.item()))

        wandb.run.summary["finalPSNR"] = psnr
        print(f"Test PSNR: {psnr}")
        if config.plot_fit:
            # plot results and log them
            plot_fitted_kernel(
                output.detach().cpu().squeeze(),
                f.detach().cpu().squeeze(),
                loss,
                config,
                plot_gt=False,
            )
        if config.save_fit:
            exp_dir = os.path.join("results", config.exp_name)
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            # plot results and log them
            plot_fitted_kernel(
                output.detach().cpu().squeeze(),
                f.detach().cpu().squeeze(),
                loss,
                config,
                plot_gt=False,
                save=os.path.join(
                    exp_dir, f"{config.model}_freq{config.freq}_order{config.order}"
                ),
            )
        # -----------------
        wandb.finish()

        # results['mse'][i_f, j_o] = loss.item()
        # results['psnr'][i_f, j_o] = psnr
        results["mse"][i_f] = loss.item()
        results["psnr"][i_f] = psnr
        # results['mse'][j_o] = loss.item()
        # results['psnr'][j_o] = psnr

    print(results)

    with open(os.path.join("results", config.exp_name + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Freq"])
        writer.writerow(map(lambda x: f"{x:.2f}", freqs.numpy()))
        # writer.writerow(['Order'])
        # writer.writerow(map(lambda x: f"{x:.2f}", orders.numpy()))
        writer.writerow(["MSE"])
        writer.writerow(map(lambda x: f"{x:.8f}", results["mse"].numpy()))
        # for i_o in range(results['mse'].shape[0]):
        #     writer.writerow(map(lambda x: f"{x:.8f}", results['mse'][i_o].numpy()))
        writer.writerow(["PSNR"])
        writer.writerow(map(lambda x: f"{x:.8f}", results["psnr"].numpy()))
        # for i_o in range(results['psnr'].shape[0]):
        #     writer.writerow(map(lambda x: f"{x:.8f}", results['psnr'][i_o].numpy()))


def get_model(config):
    out_channels = config.no_images
    if config.function == "AlexNet":
        out_channels = 1
    elif config.function == "CIFAR":
        out_channels = config.no_images
    elif config.function == "Kodak":
        out_channels = 3
    # Load the model: The model is always equal to a continuous kernel
    if config.model == "MAGNet":
        model = ckconv.nn.ck.MAGNet(
            dim_linear=2,
            hidden_channels=config.no_hidden,
            out_channels=out_channels,
            no_layers=config.no_layers,
            steerable=config.steerable,
            input_scale=256.0,
            weight_scale=1.0,
            alpha=6.0,
            beta=1.0,
            init_spatial_value=1.0,
            bias=True,
            bias_init="none",
        )
    elif config.model == "Gabor":
        model = ckconv.nn.ck.GaborNet(
            dim_linear=2,
            hidden_channels=config.no_hidden,
            out_channels=out_channels,
            no_layers=config.no_layers,
            bias=True,
            bias_init="none",
            input_scale=256.0,
            weight_scale=1.0,
            alpha=6.0,
            beta=1.0,
            init_spatial_value=1.0,
        )
    elif config.model == "Fourier":
        model = ckconv.nn.ck.FourierNet(
            dim_linear=2,
            hidden_channels=config.no_hidden,
            out_channels=out_channels,
            no_layers=config.no_layers,
            bias=True,
            bias_init="none",
            input_scale=256.0,
            weight_scale=1.0,
        )
    elif config.model == "SIREN":
        model = ckconv.nn.ck.SIREN(
            dim_linear=2,
            out_channels=out_channels,
            hidden_channels=config.no_hidden,
            weight_norm=config.weight_norm,
            no_layers=config.no_layers,
            bias=True,
            bias_init="none",
            omega_0=config.omega_0,
            learn_omega_0=config.learn_omega_0,
        )
    elif config.model == "SRF":
        model = Srf_layer_shared(
            in_channels=1,
            out_channels=1,
            init_k=config.init_k,
            init_order=config.order,
            init_scale=0.0,
            learn_sigma=True,
            use_cuda=False,
            groups=1,
            scale_sigma=0.0,
        )
    else:
        raise NotImplementedError(f"Model type {config.model} not found.")

    no_params = ckconv.utils.num_params(model)
    print("Number of parameters:", no_params)
    wandb.run.summary["no_params"] = no_params

    return model


def plot_fitted_kernel(output, f, loss, config, plot_gt, iter=None, save=None):
    if config.function == "Kodak":

        psnr = ckconv.utils.psnr(f, torch.clamp(output, 0.0, 1.0))

        if config.debug:

            fig, axs = plt.subplots(2)

            fig.suptitle(
                "Comparison function and fitted kernel.\n Loss: {:.4e}, PSNR: {:.4e}".format(
                    loss.item(), psnr
                )
            )
            axs[0].set_title("Ground truth")
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].imshow(f.permute(1, 2, 0).numpy())

            axs[1].set_title("Fitted kernel")
            axs[1].imshow(output.permute(1, 2, 0).numpy())
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            plt.show()

        # Save gt and aproximation on wandb.
        if plot_gt:
            fig = plt.figure(dpi=200)
            ax = fig.add_subplot(111)
            ax.imshow(f.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            # plt.savefig("{}.png".format(config.function), dpi=300)
            wandb.log({"ground_truth": wandb.Image(plt)}, step=iter)
            plt.close()

        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)
        ax.imshow(output.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        fig.text(
            0.99,
            0.015,
            "MSE: {:.3e}, PNSR: {:.2f}".format(loss.item(), psnr),
            verticalalignment="bottom",
            horizontalalignment="right",
            transform=ax.transAxes,
            color="Black",
            fontsize=9,
            weight="roman",
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 4},
        )
        plt.tight_layout()
        # plt.savefig("{}_{}.png".format(config.function, config.padding), dpi=300)
        wandb.log({"approximation": wandb.Image(plt)}, step=iter)
        if config.debug:
            plt.show()
        plt.close()

        return

    elif config.function == "CameraMan":

        psnr = ckconv.utils.psnr(f, torch.clamp(output, 0.0, 1.0))

        if config.debug:

            fig, axs = plt.subplots(2)

            fig.suptitle(
                "Comparison function and fitted kernel.\n Loss: {:.4e}, PSNR: {:.4e}".format(
                    loss.item(), psnr
                )
            )
            axs[0].set_title("Ground truth")
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].imshow(f.squeeze().numpy())

            axs[1].set_title("Fitted kernel")
            axs[1].imshow(output.squeeze().numpy())
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            plt.show()

    else:

        psnr = ckconv.utils.psnr(f, torch.clamp(output, 0.0, 1.0))

        if config.debug:
            fig, axs = plt.subplots(2)

            fig.suptitle(
                "Comparison function and fitted kernel.\n Loss: {:.4e}, PSNR: {:.4e}".format(
                    loss.item(), psnr
                )
            )
            matplotlib.rcParams["figure.figsize"] = [3, 4]

            axs[0].set_title("Ground truth")
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].imshow(f.numpy())

            axs[1].set_title("Fitted kernel")
            axs[1].imshow(output.numpy())
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            if save is not None:
                plt.savefig(save + ".pdf")
            else:
                plt.show()

            if save is not None:
                matplotlib.rcParams["figure.figsize"] = [3, 3]
                fig, axs = plt.subplots(1)
                axs.imshow(f.numpy())
                axs.set_xticks([])
                axs.set_yticks([])
                fig.tight_layout()
                fig.savefig(save + "_gt.pdf")

                fig, axs = plt.subplots(1)
                axs.imshow(output.numpy())
                axs.set_xticks([])
                axs.set_yticks([])
                fig.tight_layout()
                fig.savefig(save + "_pred.pdf")

        # fig, axs = plt.subplots(2, limits)
        # fig.suptitle(
        #     "Comparison function and fitted kernel. Loss: {:.4e}".format(loss.item())
        # )
        #
        # axs[0, 0].set_title("Ground truth")
        # for i in range(limits):
        #     axs[0, i].set_xticks([])
        #     axs[0, i].set_yticks([])
        #     axs[0, i].imshow(f[i])
        #
        # axs[1, 0].set_title("Fitted kernel")
        # for i in range(limits):
        #     axs[1, i].imshow(output[i], label="fitted kernel")
        #     axs[1, i].set_xticks([])
        #     axs[1, i].set_yticks([])
        # axs[1, 0].text(
        #     0.96,
        #     0.035,
        #     "Loss: {:.3e}".format(loss.item()),
        #     verticalalignment="bottom",
        #     horizontalalignment="right",
        #     transform=axs[1, 0].transAxes,
        #     color="Black",
        #     fontsize=9,
        #     weight="roman",
        #     family="monospace",
        #     bbox={"facecolor": "white", "alpha": 0.9, "pad": 4},
        # )
        # plt.show()

        # plt.savefig(
        #     "{}_{}_{}.png".format(
        #         config.function,
        #         config.kernelnet_activation_function,
        #         config.comment,
        #     ),
        #     dpi=300,
        # )
        # wandb.log({"fitted_kernel": wandb.Image(plt)})
        # plt.show()

        # Differences
        # plt.figure()
        # plt.imshow(f - output)
        # plt.xticks([])
        # plt.title("Difference (f - output). Loss: {:.4e}".format(loss.item()))
        # plt.tight_layout()
        # #wandb.log({"diff_fit_gt": wandb.Image(plt)})
        # plt.show()


def evaluate(model, x, f, config):
    # Check the fitting
    # -----------------
    model.eval()
    with torch.no_grad():
        output = model(x)
        if config.padding == 0:
            loss = torch.nn.functional.mse_loss(output, f)
            # log
            psnr = ckconv.utils.psnr(f, torch.clamp(output, 0.0, 1.0))
        else:
            loss = torch.nn.functional.mse_loss(
                output[
                    :,
                    :,
                    config.padding : -config.padding,
                    config.padding : -config.padding,
                ],
                f[
                    :,
                    :,
                    config.padding : -config.padding,
                    config.padding : -config.padding,
                ],
            )
            # log
            psnr = ckconv.utils.psnr(
                f[
                    :,
                    :,
                    config.padding : -config.padding,
                    config.padding : -config.padding,
                ],
                torch.clamp(
                    output[
                        :,
                        :,
                        config.padding : -config.padding,
                        config.padding : -config.padding,
                    ],
                    0.0,
                    1.0,
                ),
            )
    return loss, psnr


def save_model_to_wandb(model, name="model", epoch=None):
    filename = f"{name}_{epoch}.pt"
    path = os.path.join(wandb.run.dir, filename)

    torch.save(
        {
            "model": model.state_dict(),
        },
        path,
    )
    # Call wandb to save the object, syncing it directly
    wandb.save(path)


def plot_target(f):
    plt.figure()
    plt.imshow(f.detach().cpu()[0])
    plt.show()


if __name__ == "__main__":
    app.run(main)
