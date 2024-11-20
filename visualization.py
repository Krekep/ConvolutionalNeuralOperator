import random

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
wave_gauss_nc = netCDF4.Dataset("nc_data/res/Wave-Gauss.nc")
# initial_condition = netCDF4.Dataset("nc_data/c_0.nc")
print(wave_gauss_nc)
print(wave_gauss_nc.variables.keys())

sample_num = random.randint(0, 10000)
time_steps = 15
t1 = random.randint(0, 15)
t2 = random.randint(t1, 15)
time = (t2 - t1) / time_steps

inputs = torch.tensor(wave_gauss_nc["solution"][sample_num, t1]).reshape(1, 128, 128)
inputs_condition = torch.tensor(wave_gauss_nc["c"][sample_num]).reshape(1, 128, 128)
time_delta = torch.ones(1, 128, 128).type(torch.float32) * time

inputs = torch.cat([inputs, inputs_condition, time_delta], dim=0).unsqueeze(dim=0)

labels = torch.tensor(wave_gauss_nc["solution"][sample_num, t2]).reshape(1, 128, 128)
labels = torch.cat([labels, inputs_condition], dim=0).unsqueeze(dim=0)

prediction: torch.Tensor = cno.forward(inputs, torch.tensor(4))
prediction = prediction.detach()

fig, ax = plt.subplots(nrows=2, ncols=3)

ax[0, 0].imshow(inputs[0][0])
ax[0, 0].set_title(f"Input at time {t1}")
ax[1, 0].imshow(inputs_condition[0])
ax[1, 0].set_title(f"Initial conditions")

ax[0, 1].imshow(prediction[0][0])
ax[0, 1].set_title(f"Predicted at time {t2}")
ax[1, 1].imshow(prediction[0][1])
ax[1, 1].set_title(f"Predicted conditions")

ax[0, 2].imshow(labels[0][0])
ax[0, 2].set_title(f"True at time {t2}")
ax[1, 2].imshow(labels[0][1])
ax[1, 2].set_title(f"Initial conditions")

plt.show()
