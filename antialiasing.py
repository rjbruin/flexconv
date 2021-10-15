import torch
import math

import ckconv


def regularize_gabornet(
    model, kernel_size, target, fn, method, factor, gauss_stddevs=1.0, gauss_factor=0.5
):
    # if method != "summed":
    #     raise NotImplementedError()

    # Collect frequency terms to be regularized from all FlexConv modules
    modules = get_flexconv_modules(model)
    magnet_freqs = []
    mask_freqs = []
    masks = False
    for module in modules:
        module_magnet_freqs, module_mask_freq = gabor_layer_frequencies(
            module, target, method, gauss_stddevs=gauss_stddevs
        )
        magnet_freqs.append(module_magnet_freqs)
        if module_mask_freq is not None:
            mask_freqs.append(module_mask_freq)
    magnet_freqs = torch.stack(magnet_freqs)
    if len(mask_freqs) > 0:
        masks = True
        mask_freqs = torch.stack(mask_freqs)

    if method == "summed":
        # Regularize sum of all filters together, per layer
        magnet_freqs = torch.sum(magnet_freqs, 1)
        if masks:
            flexconv_freqs = magnet_freqs + mask_freqs
        else:
            flexconv_freqs = magnet_freqs

        nyquist_freq = torch.ones_like(flexconv_freqs) * nyquist_frequency(kernel_size)

    elif method == "together" and target == "gabor":
        if masks:
            raise NotImplementedError()
        else:
            flexconv_freqs = magnet_freqs

        # Divide Nyquist frequency by amount of filters in each layer
        nyquist_freq = torch.ones_like(flexconv_freqs) * nyquist_frequency(kernel_size)
        nyquist_freq = nyquist_freq / nyquist_freq.shape[1]

    elif method in ["together", "together+mask"] and target == "gabor+mask":
        if masks:
            # Distributing single mask over all filters
            n_filters = magnet_freqs.shape[1]
            mask_freqs = mask_freqs.unsqueeze(1).repeat([1, n_filters]) / torch.tensor(
                n_filters, dtype=torch.float32
            )
            flexconv_freqs = magnet_freqs + mask_freqs
        else:
            raise NotImplementedError()

        # Divide Nyquist frequency by amount of filters in each layer
        nyquist_freq = torch.ones_like(flexconv_freqs) * nyquist_frequency(kernel_size)
        nyquist_freq = nyquist_freq / nyquist_freq.shape[1]

    # if method == "distributed":
    #     # Further divide Nyquist freq between sines and gausses
    #     nyquist_freq[:, :, 0] = nyquist_freq[:, :, 0] * (1.0 - gauss_factor)
    #     nyquist_freq[:, :, 1] = nyquist_freq[:, :, 1] * gauss_factor

    if fn == "l2_relu":
        # L2 ReLU
        return factor * l2_relu(flexconv_freqs, nyquist_freq)
    elif fn == "offset_elu":
        # L1 ELU with offset and scale to approximate L2 ReLU
        return factor * offset_elu(flexconv_freqs, nyquist_freq, 4.0, 5.0)
    else:
        raise NotImplementedError(f"regularization function {fn}")


def nyquist_frequency(kernel_size):
    # Nyquist frequency = samples per X x 1/2 (for rate to freq)
    return float((kernel_size - 1.0) / 2.0) * 0.5


def freq_effect(gamma, stddevs=1.0):
    return gamma * stddevs / (2.0 * torch.tensor(math.pi))


def gabor_layer_frequencies(module, target, method, gauss_stddevs=1.0):
    n_filters = len(module.Kernel.filters)
    # If we are using distributed regularization, we have two terms for each
    # filter: the sine term and the Gaussian term
    # n_terms = 2 if target == "gabor" and method == "distributed" else 1

    freqs = torch.zeros((n_filters), dtype=torch.float32)
    for i, f in enumerate(module.Kernel.filters):
        if target == "sines":
            # All units are in Hz, not radians
            freqs[i] = torch.max(torch.absolute(f.linear.weight)) / (
                2.0 * torch.tensor(math.pi)
            )
        elif target == "gausses":
            gausses = freq_effect(f.gamma, stddevs=gauss_stddevs)
            freqs[i] = torch.max(torch.absolute(gausses))
        elif target == "gabor" or target == "gabor+mask":
            # All units are in Hz, not radians
            sines = torch.absolute(f.linear.weight / (2.0 * torch.tensor(math.pi)))
            gausses = torch.absolute(freq_effect(f.gamma, stddevs=gauss_stddevs))
            if method in ["together", "summed", "together+mask"]:
                combined = sines + gausses
                freqs[i] = torch.max(combined)
            # elif method == "distributed":
            #     freqs[i, 0] = torch.max(sines)
            #     freqs[i, 1] = torch.max(gausses)
            else:
                raise NotImplementedError(f"method {method}")
        else:
            raise NotImplementedError(f"target {target}")

    mask_freq = None
    if target == "gabor+mask":
        # Mask effect = max(x_gamma,y_gamma) where each is inverse of sigma
        x_mask_gamma = 1.0 / torch.absolute(module.mask_params[0, 1]).to(freqs.device)
        y_mask_gamma = 1.0 / torch.absolute(module.mask_params[1, 1]).to(freqs.device)
        mask_gamma = torch.maximum(x_mask_gamma, y_mask_gamma)
        mask_freq = freq_effect(mask_gamma, stddevs=gauss_stddevs)

    return freqs, mask_freq


def l2_relu(x, target):
    over_freq = torch.maximum(
        torch.tensor(0.0, device=x.device),
        x - target,
    )
    return torch.sum(torch.square(over_freq))


def offset_elu(x, target, offset, scale):
    over_freq = x - target
    condition = over_freq > offset
    elu = torch.where(
        condition, over_freq - offset + 1.0, torch.exp(over_freq - offset)
    )
    elu *= scale
    return torch.sum(elu)


def get_flexconv_modules(model):
    modules = []
    for m in model.modules():
        if isinstance(m, ckconv.nn.FlexConv):
            modules.append(m)
    return modules


def get_gabornet_summaries(model, target, method):
    if target == "gabor":
        targets = ["sines", "gausses", "gabor"]
    elif target == "gabor+mask":
        targets = ["sines", "gausses", "gabor+mask"]
    else:
        targets = [target]

    stats = {}
    modules = get_flexconv_modules(model)
    module_mask_freqs = torch.zeros((len(modules)), dtype=torch.float32)

    for t in targets:
        module_magnet_freqs = torch.zeros((len(modules)), dtype=torch.float32)
        for i, module in enumerate(modules):
            magnet_freqs, mask_freq = gabor_layer_frequencies(module, t, method)
            module_magnet_freqs[i] = torch.sum(magnet_freqs)
            stats[f"{t}_freq_{i}"] = module_magnet_freqs[i]
            if t == "gabor+mask":
                module_mask_freqs[i] = mask_freq
                stats[f"mask_freq_{i}"] = module_mask_freqs[i]
        stats[f"{t}_freq_mean"] = torch.mean(module_magnet_freqs)
        stats[f"{t}_freq_std"] = torch.std(module_magnet_freqs)

    stats[f"mask_freq_mean"] = torch.mean(module_mask_freqs)
    stats[f"mask_freq_std"] = torch.std(module_mask_freqs)

    return stats
