training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 4,
    "batch_size": 16,
    "exp": 1,  # Do we use L1 or L2 errors? Default: L1
    "training_samples": 256,  # How many training samples?
}
model_architecture_ = {
    # Parameters to be chosen with model selection:
    "N_layers": 3,  # Number of (D) & (U) blocks
    "channel_multiplier": 32,  # Parameter d_e (how the number of channels changes)
    "N_res": 4,  # Number of (R) blocks in the middle networs.
    "N_res_neck": 6,  # Number of (R) blocks in the BN
    # Other parameters:
    "in_size": 84,  # Resolution of the computational grid
    "retrain": 4,  # Random seed
    "kernel_size": 3,  # Kernel size.
    "FourierF": 0,  # Number of Fourier Features in the input channels. Default is 0.
    "activation": "cno_lrelu",  # cno_lrelu or cno_lrelu_torch or lrelu or
    # Filter properties:
    "cutoff_den": 2.0001,  # Cutoff parameter.
    "lrelu_upsampling": 2,  # Coefficient N_{\sigma}. Default is 2.
    "half_width_mult": 0.8,  # Coefficient c_h. Default is 1
    "filter_size": 6,  # 2xfilter_size is the number of taps N_{tap}. Default is 6.
    "radial_filter": 0,  # Is the filter radially symmetric? Default is 0 - NO.
}
