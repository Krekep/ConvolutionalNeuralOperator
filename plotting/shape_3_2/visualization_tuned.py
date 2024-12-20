import random
import netCDF4
import torch
import matplotlib.pyplot as plt
from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model
from visualization import create_frames
from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)

which_example = "wave_gauss"
folder = f"../../TrainedModels/fine_tuned"
label = "2"

cno, loader_dict = load_model(
    folder=folder,
    which_example=which_example,
    steps=11 if which_example == "piezo_conductivity" else 7,
    in_dim=3,
    out_dim=2,
    label=label,
    fine_tuned=True,
    fine_tuned_kwargs={
        "diff_embedding": True,
        "old_in_dim": 5,
        "new_in_dim": 3,
        "new_out_dim": 2,
    },
)
wave_gauss_nc = netCDF4.Dataset("../../nc_data/res/Wave-Gauss.nc")
print(wave_gauss_nc)
print(wave_gauss_nc.variables.keys())

constants = {
    "mean": 0.0334376316,
    "std": 0.1171879068,
    "mean_c": 2618.4593933,
    "std_c": 601.51658913,
    "time": 15,
}
create_frames(
    cno, wave_gauss_nc, constants, gif_name=f"cno-tuned-{which_example}_{label}"
)
