## FlexConv: Continuous Kernel Convolutions with Differentiable Kernel Sizes

This repository contains the source code accompanying the paper:

[FlexConv: Continuous Kernel Convolutions with Differentiable Kernel Sizes](https://arxiv.org/abs/2102.02611)  [[Slides]](#) <br/>**[David W. Romero*](https://www.davidromero.ml/), [Robert-Jan Bruintjes*](https://rjbruin.github.io), [Jakub M. Tomczak](https://jmtomczak.github.io/), [Erik J. Bekkers](https://erikbekkers.bitbucket.io/), [Mark Hoogendoorn](https://www.cs.vu.nl/~mhoogen/) & [Jan C. van Gemert](https://jvgemert.github.io)**.

This experimental code extends [the kernel fitting experimental code by David W. Romero et al.](https://github.com/dwromero/ckconv). We add new kernel generators covered in our paper, specifically MFNs with Fourier and Gabor features, as well as our proposed MAGNet kernel generator.

_The rest of this description is reproduced from [the original codebase](https://github.com/dwromero/ckconv)._

### Fitting Convolutional Kernels via MLPs

In this folder, you can find the code used in our experiments to fit implicit neural representations to several different target functions.

#### Functionality:

Several functions can be found in `functions.py`. Some functions may receive some parameters as input. For example,
a Gaussian might be parameterized via its mean and its variance. This parameters are controlled via `argparse` and follow
the dictionary of parameters found in `config.py`.

Other parameters in `config.py` are used to parameterized the MLP used in the experiment. For instance, define the
nonlinearity used.

An example is, for instance:
```
python fit_function.py --config.function SinusChirp --config.min -15.0 --config.max 0.0 --config.no_samples 1000 --config.optim Adam --config.lr 1e-4 --config.no_iterations 20000 --config.seed 0 --config.device cuda --config.kernelnet_norm_type LayerNorm --config.kernelnet_activation_function Sine --config.kernelnet_no_hidden 32 --config.padding 0 --config.kernelnet_omega_0 1000.0
```

This call will fit a SinusShirp defined between -15 and 0 sampled at 1000 positions. The network used to approximate that function is given by a 3-layer MLP with 32 hidden units, Sine nonlinearities and omega_0 = 1000.