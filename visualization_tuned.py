import random
import netCDF4
import torch
import matplotlib.pyplot as plt
from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model
from visualization import create_frames


which_example = "wave_gauss"
folder = f"TrainedModels/fine_tuned"
loader_dict = _load_dict(
    files=[folder + "/training_properties.txt", folder + "/net_architecture.txt"],
    which_example=which_example,
)
print(loader_dict)
cno = _initialize_model(
    loader_dict,
    diff_embedding=True,
    old_in_dim=5,
    new_in_dim=3,
    new_out_dim=2,
)
wave_gauss_nc = netCDF4.Dataset("nc_data/res/Wave-Gauss.nc")
print(wave_gauss_nc)
print(wave_gauss_nc.variables.keys())

create_frames(cno, wave_gauss_nc, "cno-tuned-gauss.gif")
