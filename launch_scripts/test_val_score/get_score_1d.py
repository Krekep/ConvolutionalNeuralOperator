import math

import netCDF4
import numpy as np
import json
import torch

import sys

sys.path.append("../../CNO2d_time_dependent_and_foundation_model")
from test_and_fine_tune_utils.fine_tune_utils import load_model

path_to_folder = "../data_process"
dataset_nc = netCDF4.Dataset(f"{path_to_folder}/navier_stokes_no_condition.nc")
solution = dataset_nc["solution"]  # (4096, 20, 128, 128)

constants = {  # dataset solution mean, std and initial conditions constants
    "mean": 1.929536892930628e-06,
    "std": 1.131550669670105,
    "time": 20,
}

n_max = 1000  # samples in dataset
n_test = 1  # number of test samples
n_val = 100  # number of training samples
start = n_max - n_test - n_val  # where training samples ends
length = n_val + n_test

for label in ["880"]:  # model labels
    which_example = "navier_stokes_no_condition"
    variant = "1"
    model, loader_dict = load_model(
        folder=f"../../TrainedModels/Time_CNO_{which_example}_{variant}",
        which_example=which_example,
        steps=10,
        in_dim=2,
        out_dim=1,
        label=label,
        in_size=64,
    )

    sol_copy = np.copy(solution)

    end_time = constants["time"]

    best_samples = []

    times = [
        (
            torch.ones(1, 64, 64, requires_grad=False).type(torch.float32)
            * (t - 0)  # (t - start_time) / end_time  --- [0; 1]
            / end_time,
            t,  # time index --- [0, 20]
        )
        for t in range(0, end_time)
    ]

    loss = torch.nn.L1Loss()
    for sample in range(start, start + length):
        # if (sample - start) % 10 == 0:
        #     print(sample, best_samples[-3:])
        mae = 0
        start_time = 0

        # input is (function(time=i), condition, time_delta), output is (function(time=j), condition), j >= i
        inputs_function = torch.tensor(
            sol_copy[sample, start_time], requires_grad=False
        ).reshape(1, 64, 64)
        inputs_function = (inputs_function - constants["mean"]) / constants["std"]

        for t, t_idx in times:
            time = (t_idx - start_time) / end_time
            inputs = torch.cat([inputs_function, t], dim=0).unsqueeze(dim=0)

            labels = torch.tensor(sol_copy[sample, t_idx], requires_grad=False).reshape(
                1, 64, 64
            )
            labels = (labels - constants["mean"]) / constants["std"]
            labels = torch.cat([labels], dim=0)
            labels = labels.unsqueeze(dim=0)

            predicted = model.forward(
                inputs,
                torch.tensor(
                    time, requires_grad=False
                ),  # model take input and scalar time
            ).detach()

            all_loss = torch.nn.L1Loss()(predicted, labels)
            mae += all_loss

        best_samples.append((mae, sample))
        torch.clear_autocast_cache()

    best_samples.sort()
    print("Label", label)
    print("Best", best_samples[:5])
    print("Worst", best_samples[-5:])
    with open(f"{which_example}_validation_score_{variant}_{label}.txt", "w") as f:
        f.write(
            ", ".join(
                map(
                    lambda x: str(round(x[0].item(), 4)) + ", " + str(x[1]),
                    best_samples,
                )
            )
            + "\n"
        )
    print("*********")
