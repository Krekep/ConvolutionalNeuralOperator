import netCDF4
import torch

from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)
import matplotlib.pyplot as plt

which_example = "wave_gauss"
model = load_model(
    folder=f"TrainedModels/Time_CNO_{which_example}_1",
    which_example=which_example,
    in_dim=3,
    out_dim=2,
    label="794",
)
cno = model[0]
print(model[1].keys())
nc = netCDF4.Dataset("nc_data/res/Wave-Gauss.nc")
print(nc)
print(nc.variables.keys())

samples = []
for i in range(1):
    samples.append(torch.tensor(nc["solution"][256 + i][:3]))
sample = torch.stack(samples)

prediction: torch.Tensor = cno.forward(sample, torch.tensor(4))
print(prediction)
print(prediction.shape)
