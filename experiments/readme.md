# Experiments

We provide the commands used to run the experiments published in the paper. For each experiment we provide a single command. For experiments where results are reported over multiple runs one should use incremental integer seeds starting at zero to reproduce the original results. For example, for an experiment with three runs we used `seed=0`, `seed=1` and `seed=2`.

For selected models, we provide checkpoints of trained models. For the aliasing regularization models we provide checkpoints obtained after training on the source resolution as well as after fine-tuning on the target resolution. Checkpoints for any of the experiments listed below can be made available upon request. To make such a request, please file an issue in this repository.

Please note that due to randomness in certain PyTorch operations on CUDA, it may not be possible to reproduce certain results with high precision. Please see [PyTorch's manual on deterministic behavior](https://pytorch.org/docs/stable/notes/randomness.html) for more details, as well as `run_experiments.py::set_manual_seed()` for specifications on how we seed our experiments.

## Sequence classification

*TODO(dwromero)*

## Image classification

### CIFAR-10 (Table 3)

FlexNet-7. [[Checkpoint]](checkpoints/cifar10-flexnet-7-seed2.pt) (seed 2, acc. 92.31%)
```
python run_experiment.py conv.horizon=33 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 "summary=[64, 3, 32, 32]" seed=0
```

FlexNet-7 with convolutions (`k=3`).
```
python run_experiment.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" conv.type=Conv conv.horizon=3 dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam train.scheduler=cosine train.weight_decay=0 "summary=[64, 3, 32, 32]" seed=0
```

FlexNet-7 with convolutions (`k=33`).
```
python run_experiment.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" conv.type=Conv conv.horizon=33 dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam train.scheduler=cosine train.weight_decay=0 "summary=[64, 3, 32, 32]" seed=0
```

FlexNet-7 with N-Jet layers (Pintea et al. 2021)
```
python run_experiment.py conv.type=SRF dataset=CIFAR10 device=cuda "net.block_width_factors=[1, 2, 1.5, 3, 2, 2]" net.dropout=0.2 net.dropout_in=0 net.no_blocks=7 net.no_hidden=38 net.norm=BatchNorm net.type=ResNet seed=0 train.augment=resnet train.batch_size=64 train.epochs=350 train.lr=0.01 train.mask_params_lr_factor=0.1 train.optimizer=Adam train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05
```

FlexNet-7 with SIREN kernels and without FlexConv mask (aka CKCNN).
```
python run_experiment.py conv.horizon=33 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=CKConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=SIREN kernel.omega_0=30.0 kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.001 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=1
```

FlexNet-7 with MAGNet kernels and without FlexConv mask (aka CKCNN-MAGNet).
```
python run_experiment.py conv.horizon=33 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=CKConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=1
```

FlexNet-7 with SIREN kernels.
```
python run_experiment.py conv.horizon=33 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=SIREN kernel.omega_0=30.0 kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.001 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=1
```

FlexNet-7 with MFN-Gabor kernels.
```
python run_experiment.py conv.horizon=33 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=Gabor kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=1
```

### ImageNet-32 (Table 6)

FlexNet-5. [[Checkpoint]](checkpoints/imagenet32-flexnet-5-seed1.pt) (seed 1, 25.31%)

```
python run_experiment.py conv.horizon=33 conv.type=FlexConv dataset=Imagenet32 dataset_params.from_cache=True dataset_params.in_ram=True dataset_params.root=<link-to-imagenet-32-raw-data> device=cuda kernel.input_scale=25.6 kernel.no_hidden=32 kernel.no_layers=3 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.temperature=15 mask.type=gaussian "net.block_width_factors=[1.0, 1, 1.5, 2, 2.0, 2]" net.dropout=0.2 net.dropout_in=0 net.no_blocks=5 net.no_hidden=22 net.norm=BatchNorm net.type=ResNet train.augment=resnet train.batch_size=2048 train.epochs=350 train.lr=0.01 train.mask_params_lr_factor=0.1 train.optimizer=Adam train.report_top5_acc=True train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05 seed=0
```

ResNet-32. This model was run on an older version of the experimental code. The command given would reproduce the original results.
```
python run_experiment.py train.augment=resnet train.batch_size=64 device=cuda dataset=CIFAR10 train.epochs=350 train.lr=0.001 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=CIFARResNet32 train.optimizer=Adam train.scheduler=cosine train.weight_decay=0 seed=0
```

### MNIST (Table 8)

FlexNet-7.
```
python run_experiment.py conv.horizon=29 train.augment=None train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=MNIST device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=350 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 "summary=[64, 1, 28, 28]" seed=1
```

### STL-10 (Table 9)

FlexNet-7.
```
python run_experiment.py conv.horizon=97 train.augment=None train.batch_size=256 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=STL10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 summary="[64, 3, 96, 96]" seed=3
```

## Aliasing regularization

Use script `run_crossres.py` to reproduce these experiments. Use flags `cross_res.source_res` and `cross_res.target_res` to change the source and target resolution for this experiment, to reproduce Figure 5. For ImageNet-k, the only supported resolutions are `[16, 32, 64]`, as these are the only resolutions for which there are datasets available. For CIFAR-10, images will be up- or downsampled to the desired resolutions, unless the resolution is equal to the original resolution.

### CIFAR-10 (Table 4)

FlexNet-7 with aliasing regularization without FlexConv mask. [[Source checkpoint]](checkpoints/cross-res-cifar10-flexnet-7-gabor-seed2-source.pt) (seed 2, 86.61%) [[Final checkpoint]](checkpoints/cross-res-cifar10-flexnet-7-gabor-seed2-final.pt) (seed 2, 89.16%)
```
python run_crossres.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" device=cuda conv.type=FlexConv dataset=CIFAR10 net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 cross_res.finetune_epochs=100 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam kernel.regularize=True kernel.regularize_params.factor=0.1 train.scheduler=cosine cross_res.source_res=16 cross_res.target_res=32 train.weight_decay=0 kernel.regularize_params.target=gabor kernel.regularize_params.method=together kernel.regularize_params.gauss_stddevs=2.0 seed=0
```

FlexNet-7 with aliasing regularization with FlexConv mask. [[Source checkpoint]](checkpoints/cross-res-cifar10-flexnet-7-gnm-seed1-source.pt) (seed 1, 85.25%) [[Final checkpoint]](checkpoints/cross-res-cifar10-flexnet-7-gnm-seed1-final.pt) (seed 1, 87.85%)
```
python run_crossres.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" device=cuda conv.type=FlexConv dataset=CIFAR10 net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 cross_res.finetune_epochs=100 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam kernel.regularize=True kernel.regularize_params.factor=0.1 train.scheduler=cosine cross_res.source_res=16 cross_res.target_res=32 train.weight_decay=0 kernel.regularize_params.target=gabor+mask kernel.regularize_params.method=together kernel.regularize_params.gauss_stddevs=2.0 seed=0
```

ResNet-44
```
python run_crossres.py train.augment=resnet train.batch_size=64 device=cuda dataset=CIFAR10 train.epochs=100 cross_res.finetune_epochs=100 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=CIFARResNet44 train.optimizer=Adam train.scheduler=cosine cross_res.source_res=16 cross_res.target_res=32 train.weight_decay=0 seed=0
```

FlexNet-7 with convolutions (`k = 3`)
```
python run_crossres.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" conv.type=Conv conv.horizon=3 dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 cross_res.finetune_epochs=100 cross_res.source_res=16 cross_res.target_res=32 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam train.scheduler=cosine train.weight_decay=0 "summary=[64, 3, 32, 32]" seed=0
```

FlexNet-7 with convolutions (`k = 33`)
```
python run_crossres.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" conv.type=Conv conv.horizon=33 dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 cross_res.finetune_epochs=100 cross_res.source_res=16 cross_res.target_res=32 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam train.scheduler=cosine train.weight_decay=0 "summary=[64, 3, 32, 32]" seed=0
```

FlexNet-7 with N-Jet layers (Pintea et al. 2021).
```
python run_crossres.py comment=c10-cres-srf7 conv.type=SRF dataset=CIFAR10 device=cuda "net.block_width_factors=[1, 2, 1.5, 3, 2, 2]" net.dropout=0.2 net.dropout_in=0 net.no_blocks=7 net.no_hidden=38 net.norm=BatchNorm net.type=ResNet train.augment=resnet train.batch_size=64 train.epochs=100 train.lr=0.01 train.mask_params_lr_factor=0.1 train.optimizer=Adam train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05 cross_res.source_res=16 cross_res.target_res=32 cross_res.finetune_epochs=100 seed=0
```

FlexNet-7 with SIREN kernels and without FlexConv mask (aka "CKCNN").
```
python run_crossres.py cross_res.source_res=16 cross_res.target_res=32 cross_res.finetune_epochs=100 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=CKConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 kernel.input_scale=25.6 kernel.type=SIREN kernel.omega_0=30.0 kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=0
```

FlexNet-7 with SIREN kernels.
```
python run_crossres.py cross_res.source_res=16 cross_res.target_res=32 cross_res.finetune_epochs=100 train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" net.no_blocks=7 net.no_hidden=24 conv.type=FlexConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 kernel.input_scale=25.6 kernel.type=SIREN kernel.omega_0=30.0 kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine train.weight_decay=0 seed=0
```

FlexNet-7 without aliasing regularization.
```
python run_crossres.py train.augment=resnet train.batch_size=64 "net.block_width_factors=[1.0, 2, 1.5, 3, 2.0, 2]" conv.type=FlexConv dataset=CIFAR10 device=cuda net.dropout=0.2 net.dropout_in=0 mask.dynamic_cropping=True train.epochs=100 cross_res.finetune_epochs=100 kernel.input_scale=25.6 kernel.type=MAGNet kernel.no_hidden=32 kernel.no_layers=3 train.lr=0.01 train.scheduler_params.warmup_epochs=5 train.mask_params_lr_factor=0.1 mask.init_value=0.075 mask.temperature=15.0 mask.type=gaussian net.type=ResNet net.no_blocks=7 net.no_hidden=24 net.norm=BatchNorm train.optimizer=Adam kernel.regularize=False train.scheduler=cosine cross_res.source_res=16 cross_res.target_res=32 train.weight_decay=0 seed=0
```

### Imagenet-k (Table 7)

FlexNet-5. [[Source checkpoint]](checkpoints/cross-res-imagenetk-flexnet-5-gabor-seed1-source.pt) (seed 1, 14.92%) [[Final checkpoint]](checkpoints/cross-res-imagenetk-flexnet-5-gabor-seed1-final.pt) (seed 1, 25.09%)
```
python run_crossres.py cross_res.source_res=16 cross_res.target_res=32 conv.type=FlexConv dataset=Imagenet-k dataset_params.from_cache=True dataset_params.in_ram=True dataset_params.root=/home/nfs/robertjanbruin/CV-Datasets/Imagenet-k/raw-data device=cuda kernel.input_scale=25.6 kernel.no_hidden=32 kernel.no_layers=3 kernel.regularize=True kernel.regularize_params.factor=0.1 kernel.regularize_params.gauss_stddevs=2 kernel.type=MAGNet mask.dynamic_cropping=True mask.init_value=0.075 mask.temperature=15 mask.type=gaussian "net.block_width_factors=[1.0, 1, 1.5, 2, 2.0, 2]" net.dropout=0.2 net.dropout_in=0 net.no_blocks=5 net.no_hidden=22 net.norm=BatchNorm net.type=ResNet train.augment=resnet train.batch_size=2048 train.epochs=100 cross_res.finetune_epochs=100 train.lr=0.01 train.mask_params_lr_factor=0.1 train.optimizer=Adam train.report_top5_acc=True train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05 kernel.regularize_params.target=gabor kernel.regularize_params.method=together seed=0
```

ResNet-32.
```
python run_crossres.py cross_res.source_res=16 cross_res.target_res=32 dataset=Imagenet-k dataset_params.from_cache=True dataset_params.in_ram=True dataset_params.root=/home/nfs/robertjanbruin/CV-Datasets/Imagenet-k/raw-data device=cuda net.type=CIFARResNet32 train.augment=resnet train.batch_size=2048 train.epochs=100 cross_res.finetune_epochs=100 train.lr=0.01 train.optimizer=Adam train.report_top5_acc=True train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05 seed=0
```

FlexNet-5 with N-Jet layers (Pintea et al. 2021).
```
python run_crossres.py cross_res.source_res=16 cross_res.target_res=32 conv.type=SRF dataset=Imagenet-k dataset_params.from_cache=True dataset_params.in_ram=True dataset_params.root=/home/nfs/robertjanbruin/CV-Datasets/Imagenet-k/raw-data device=cuda "net.block_width_factors=[1.0, 1, 1.5, 2, 2.0, 2]" net.dropout=0.2 net.dropout_in=0 net.no_blocks=5 net.no_hidden=32 net.norm=BatchNorm net.type=ResNet train.augment=resnet train.batch_size=2048 train.epochs=100 cross_res.finetune_epochs=100 train.lr=0.01 train.mask_params_lr_factor=0.1 train.optimizer=Adam train.report_top5_acc=True train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=1e-05 seed=0
```