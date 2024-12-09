training_properties = {
    "learning_rate": 0.00075,
    "weight_decay": 1e-6,
    "scheduler_step": 1,
    "scheduler_gamma": 0.9,
    "epochs": 250,
    "batch_size": 32,
    "time_steps": 10,  # How many time steps to select?
    "dt": 1,  # What is the time step? (1 means include entire traj, 2 means taking every other step, etc.
    "training_samples": 64,  # How many training samples?
    "time_input": 1,  # Should we include time in the input channels?
    "allowed": "all",  # All2ALL (train) - all , or One2All (train) - one2all, AR training - one
    "cluster": True,  # Something internal (don't bother)
}

model_architecture_ = {
    "N_layers": 4,  # Number of (D) & (U) blocks
    "channel_multiplier": 32,  # Parameter d_e (how the number of channels changes)
    "N_res": 8,  # Number of (R) blocks in the middle networs.
    "N_res_neck": 8,  # Number of (R) blocks in the BN
    "batch_norm": 1,  # Should we use simple BN -- 1: use it? If is_time == 1, we turn it off
    "is_time": 1,  # Should we conditional BN/LN/IN?
    "nl_dim": "23",  # If yes, which norm? '23'-IN, '023'-BN, '123'-LN
    "in_size": 128,  # Resolution of the computational grid
    "activation": "cno_lrelu",  # cno_lrelu, cno_lrelu_torch or lrelu
    "is_att": False,  # Should we use attention in the bottleneck? You could add it!
    "patch_size": 1,  # ViT parameters, if is_att == True
    "dim_multiplier": 1,
    "depth": 3,
    "heads": 2,
    "dim_head_multiplier": 0.5,
    "mlp_dim_multiplier": 1.0,
    "emb_dropout": 0.0,
}
