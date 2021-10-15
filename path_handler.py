import os
import pathlib
from hydra import utils


def model_path(cfg, folder="saved"):

    root = pathlib.Path(os.path.join(utils.get_original_cwd(), folder))
    filename = f"{cfg.dataset}"

    # Dataset-specific keys
    if cfg.dataset in ["AddProblem", "CopyMemory"]:
        filename += f"_seqlen_{cfg.dataset_params.seq_length}"
        if cfg.dataset in ["CopyMemory"]:
            filename += f"_memsize_{cfg.dataset_params.memory_size}"

    elif cfg.dataset in ["MNIST"]:
        filename += "_perm_{}".format(
            cfg.permuted,
        )

    elif cfg.dataset in ["CharTrajectories", "SpeechCommands"]:
        if cfg.dataset in ["SpeechCommands"]:
            filename += "_mfcc_{}".format(
                cfg.mfcc,
            )
        if (cfg.dataset in ["SpeechCommands"] and not cfg.mfcc) or cfg.dataset in [
            "CharTrajectories"
        ]:
            filename += "_srtr_{}_drop_{}".format(
                cfg.sr_train,
                cfg.drop_rate,
            )
    filename += "_augm_{}".format(cfg.augment)

    # Model-specific keys
    filename += "_model_{}_blcks_{}_nohid_{}".format(
        cfg.model,
        cfg.no_blocks,
        cfg.no_hidden,
    )
    filename += "_kernnohid_{}_kernact_{}".format(
        cfg.kernelnet_no_hidden,
        cfg.kernelnet_activation_function,
    )
    if cfg.kernelnet_activation_function == "Sine":
        filename += "_kernomega0_{}".format(round(cfg.kernelnet_omega_0, 2))
    else:
        filename += "_kernnorm_{}".format(cfg.kernelnet_norm_type)

    # elif config.model in ["BFCNN", "TCN"]:
    #     filename += "_kernsize_{}".format(config.cnn_kernel_size)

    # Optimization arguments
    filename += "_bs_{}_optim_{}_lr_{}_ep_{}_dpin_{}_dp_{}_wd_{}_seed_{}_sched_{}_schdec_{}".format(
        cfg.batch_size,
        cfg.optimizer,
        cfg.lr,
        cfg.epochs,
        cfg.dropout_in,
        cfg.dropout,
        cfg.weight_decay,
        cfg.seed,
        cfg.scheduler,
        cfg.sched_decay_factor,
    )
    if cfg.scheduler == "plateau":
        filename += "_pat_{}".format(cfg.sched_patience)
    else:
        filename += "_schsteps_{}".format(cfg.sched_decay_steps)

    # Comment
    if cfg.comment != "":
        filename += "_comment_{}".format(cfg.comment)

    # Add correct termination
    filename += ".pt"

    # Check if directory exists and warn the user if the it exists and train is used.
    os.makedirs(root, exist_ok=True)
    path = root / filename
    cfg.path = str(path)

    if cfg.train and path.exists():
        print("WARNING! The model exists in directory and will be overwritten")
