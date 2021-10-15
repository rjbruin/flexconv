# general
import os
import datetime
import wandb
import hydra
from omegaconf import OmegaConf
from hydra import utils

# project
import trainer
import tester
from run_experiment import setup, model_and_datasets
from dataset_constructor import DATASET_RESOLUTIONS


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    imagenet_k = False
    if cfg.dataset == "Imagenet-k":
        if cfg.cross_res.source_res not in [16, 32, 64]:
            raise NotImplementedError(
                f"Imagenet-{cfg.cross_res.source_res} for source res."
            )
        if cfg.cross_res.target_res not in [16, 32, 64]:
            raise NotImplementedError(
                f"Imagenet-{cfg.cross_res.target_res} for target res."
            )
        imagenet_k = True

    #
    # SOURCE RESOLUTION
    #

    # Modify cfg for cross-res source training
    if cfg.cross_res.finetune_epochs < 1:
        raise ValueError("finetune_epochs must be more than zero.")
    if cfg.cross_res.source_res < 1 or cfg.cross_res.target_res < 1:
        raise ValueError("source_res and target_res more than zero.")
    if cfg.cross_res.source_res % 2 != 0:
        raise ValueError(f"source_res must be even, is {cfg.source_res}.")
    if cfg.cross_res.target_res % 2 != 0:
        raise ValueError(f"target_res must be even, is {cfg.target_res}.")
    if cfg.cross_res.source_res == cfg.cross_res.target_res:
        raise ValueError("source_res and target_res cannot be equal.")

    if cfg.conv.type != "Conv":
        cfg.conv.horizon = str(cfg.cross_res.source_res + 1)
    if cfg.kernel.regularize:
        cfg.kernel.regularize_params.res = cfg.cross_res.source_res + 1
        # DEBUG(rjbruin)
        cfg.kernel.regularize_params.res += cfg.kernel.regularize_params.res_offset
    if imagenet_k:
        cfg.dataset = f"Imagenet{cfg.cross_res.source_res}"
        original_root = cfg.dataset_params.root
        cfg.dataset_params.root = original_root.replace("Imagenet-k", cfg.dataset)
    else:
        cfg.cross_res.resize = f"{cfg.cross_res.source_res},{cfg.cross_res.source_res}"

    setup(cfg)
    model, dataloaders = model_and_datasets(cfg)

    # Train the model
    # Print arguments (Sanity check)
    print(cfg)
    # Train the model
    print(datetime.datetime.now())

    if cfg.train.do:
        model, optimizer, lr_scheduler = trainer.train(model, dataloaders, cfg)

        # Save model separately
        trainer.save_to_wandb(
            model, optimizer, lr_scheduler, cfg, name="final_source_res_model"
        )

    # Test model on the old resolution, with prefix "final_source_res"
    source_acc, source_top5 = tester.test(model, dataloaders["test"], cfg)
    wandb.log({f"final_source_res_accuracy_test": source_acc}, step=cfg.train.epochs)
    if cfg.train.report_top5_acc:
        wandb.log(
            {f"final_source_res_accuracy_top5_test": source_top5}, step=cfg.train.epochs
        )

    #
    # TARGET RESOLUTION
    #

    print(f"\n\nSwitching to target resolution: {cfg.cross_res.target_res}")

    # Adapt model to new resolution by changing cfg then initializing again
    cfg.train.epochs = cfg.cross_res.finetune_epochs
    if cfg.conv.type != "Conv":
        cfg.conv.horizon = str(cfg.cross_res.target_res + 1)
    if cfg.kernel.regularize:
        cfg.kernel.regularize_params.res = cfg.cross_res.target_res + 1
    if imagenet_k:
        cfg.dataset = f"Imagenet{cfg.cross_res.target_res}"
        cfg.dataset_params.root = original_root.replace("Imagenet-k", cfg.dataset)
    else:
        cfg.cross_res.resize = f"{cfg.cross_res.target_res},{cfg.cross_res.target_res}"
    # Don't need to resize if the dataset is already of that resolution
    if (
        cfg.dataset in DATASET_RESOLUTIONS
        and DATASET_RESOLUTIONS[cfg.dataset] == cfg.cross_res.target_res
    ):
        cfg.cross_res.resize = ""
    sres = float(cfg.cross_res.source_res)
    tres = float(cfg.cross_res.target_res)
    cfg.kernel.sampling_rate_norm = (sres / tres) ** 2
    if cfg.conv.type == "SRF":
        cfg.kernel.srf.scale = tres / sres
    # Use the local weights file of the previous model
    cfg.pretrained = True
    cfg.pretrained_params.filepath = os.path.join(
        wandb.run.dir, "final_source_res_model.pt"
    )

    print(cfg)

    # Don't setup again, but do reset model and datasets
    model, dataloaders = model_and_datasets(cfg)

    # Test model on the new resolution out-of-the-box
    ootb_acc, ootb_top5 = tester.test(model, dataloaders["test"], cfg)
    wandb.log({f"ootb_accuracy_test": ootb_acc}, step=cfg.train.epochs)
    if cfg.train.report_top5_acc:
        wandb.log({f"ootb_accuracy_top5_test": ootb_top5}, step=cfg.train.epochs)

    # Separately log the difference between final los-res and OOTB
    wandb.run.summary["ootb_difference"] = ootb_acc - source_acc
    if cfg.train.report_top5_acc:
        wandb.run.summary["ootb_difference_top5"] = ootb_top5 - source_top5

    # Fine-tune at target resolution
    trainer.train(model, dataloaders, cfg, epoch_start=cfg.train.epochs)


if __name__ == "__main__":
    main()
