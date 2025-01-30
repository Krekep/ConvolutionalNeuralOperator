import math

import netCDF4
import numpy as np
import json
import torch

import sys

sys.path.append("../../CNO2d_time_dependent_and_foundation_model")
from test_and_fine_tune_utils.fine_tune_utils import load_model

path_to_folder = ""
dataset_nc = netCDF4.Dataset(f"{path_to_folder}/piezo_conductivity.nc")
solution = dataset_nc["solution"]  # (4096, 20, 128, 128)
c = dataset_nc["c"]  # (4096, 128, 128)

constants = {  # dataset solution mean, std and initial conditions constants
    "mean": -0.004365722648799419,
    "std": 0.7487624287605286,
    "mean_c": -1.512402399497792e-12,
    "std_c": 1.387237325012336e-09,
    "time": 20,  # time steps in dataset
}

n_max = 4096  # samples in dataset
n_test = 240  # number of test samples
n_val = 120  # number of training samples
start = n_max - n_test - n_val  # where training samples ends
length = n_val + n_test

for label in ["760", "383"]:  # model labels
    which_example = "piezo_conductivity"
    variant = "3"
    model, loader_dict = load_model(
        folder=f"../TrainedModels/Time_CNO_{which_example}_{variant}",
        which_example=which_example,
        steps=11,
        in_dim=3,
        out_dim=2,
        label=label,
    )

    # indices in training
    f = open(
        f"../TrainedModels/Time_CNO_{which_example}_{variant}/model{label}/indices.txt",
        "r",
    )
    temp = json.load(f)

    # the indices in the dataset are different from those used in training
    indices = temp["old_keys"]
    new_indices = temp["new_keys"]
    sol_copy = np.copy(solution)
    c_copy = np.copy(c)
    print(indices[0], new_indices[0])
    sol_copy[indices] = sol_copy[new_indices]
    c_copy[indices] = c_copy[new_indices]

    end_time = constants["time"]

    best_samples = []

    times = [
        (
            torch.ones(1, 128, 128, requires_grad=False).type(torch.float32)
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
        ).reshape(1, 128, 128)
        inputs_function = (inputs_function - constants["mean"]) / constants["std"]

        inputs_condition = torch.tensor(c_copy[sample], requires_grad=False).reshape(
            1, 128, 128
        )
        inputs_condition = (inputs_condition - constants["mean_c"]) / constants["std_c"]

        for t, t_idx in times:
            time = (t_idx - start_time) / end_time
            inputs = torch.cat([inputs_function, inputs_condition, t], dim=0).unsqueeze(
                dim=0
            )

            labels = torch.tensor(sol_copy[sample, t_idx], requires_grad=False).reshape(
                1, 128, 128
            )
            labels = (labels - constants["mean"]) / constants["std"]
            labels = torch.cat([labels, inputs_condition], dim=0)
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
