import ml_collections


def get_config():
    default_config = dict(
        # General parameters
        function="",
        # Specifies the function to be approximated, e.g., SineShirp
        no_images=1,
        # Specifies the number of images that must be fitted.
        max=0.0,
        # Specifies the maximum value on which the function will be evaluated. e.g., -15.0.
        min=0.0,
        # Specifies the maximum value on which the function will be evaluated. e.g., 0.0.
        no_samples=0,
        # Specifies the number of samples that will be taken from the selected function,
        # between the min and max values.
        padding=0,
        # Specifies an amount of zero padding steps which will be concatenated at the end of
        # the sequence created by function(min, max, no_samples).
        optimizer="",
        # The optimizer to be used, e.g., Adam.
        lr=0.0,
        # The lr to be used, e.g., 0.001.
        magnitude_reg=0.0,
        # A regularization factor over the w0 * W coefficients of the layers.
        no_iterations=0,
        # The number of training iterations to be executed, e.g., 20000.
        seed=0,
        # The seed of the run. e.g., 0.
        device="",
        # The device in which the model will be deployed, e.g., cuda.
        model="SIREN",
        # Parameters of ConvKernel
        norm_type="",
        # If model == CKCNN, the normalization type to be used in the MLPs parameterizing the convolutional
        # kernels. If kernelnet_activation_function==Sine, no normalization will be used. e.g., LayerNorm.
        activation_function="",
        # If model == CKCNN, the activation function used in the MLPs parameterizing the convolutional
        # kernels. e.g., Sine.
        no_hidden=0,
        # If model == CKCNN, the number of hidden units used in the MLPs parameterizing the convolutional
        # kernels. e.g., 32.
        # Parameters of SIREN
        omega_0=0.0,
        # If model == CKCNN, kernelnet_activation_function==Sine, the value of the omega_0 parameter, e.g., 30.
        omega_0_scheme="all_same",
        # Provide a scheme for how to configure the omega_0s of each layer in the KernelNet.
        omega_0_hidden=1.0,
        # When using --omega_0_scheme=fix_hidden, provide the fixed value of omega_0
        hidden_bottleneck_factor=1.0,
        # If model == CKCNN, the bottleneck factor applied to the hidden units of all hidden layers except the first.
        weight_norm=False,
        # If model == CKCNN, specifies if the linear layers in the MLP parameterizing the convolutional
        # kernels will use weight_norm.
        no_layers=-1,
        # If model == CKCNN, specifies the numbmer of linear layers in the MLP parameterizing the convolutional
        # kernels.
        learn_omega_0=False,
        # If True, then omega_zero will be trained.
        comment="",
        # An additional comment to be added to the config.path parameter specifying where
        # the network parameters will be saved / loaded from.
        log_interval=50,
        debug=False,
        pretrained=False,
        train=True,
        sam=False,
        scheduler="",
        smooth_reg=0.0,
        sched_patience=0,
        sched_decay_factor=0.0,
        weight_decay=0.0,
        steerable=False,
        model_save_interval=500,
        image_idx=-1,
        plot_target=False,
        plot_fit=False,
        save_fit=False,
        freq=100,
        init_k=2.0,
        order=5,
        gauss_sigma=5.0,
        exp_name="debug",
    )
    default_config = ml_collections.ConfigDict(default_config)
    return default_config
