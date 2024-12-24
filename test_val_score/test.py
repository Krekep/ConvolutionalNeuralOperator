import math

import netCDF4
import numpy as np
import torch

import sys

sys.path.append("../CNO2d_time_dependent_and_foundation_model")
from test_and_fine_tune_utils.fine_tune_utils import load_model

dataset_nc = netCDF4.Dataset("../gp_data/piezo_conductivity.nc")
solution = dataset_nc["solution"]
c = dataset_nc["c"]

constants = {
    "mean": -0.004365722648799419,
    "std": 0.7487624287605286,
    "mean_c": -1.512402399497792e-12,
    "std_c": 1.387237325012336e-09,
    "time": 20,
}

# model 645
# constants = {
#     "mean": 0.020957065746188164,
#     "std": 0.9308421611785889,
#     "mean_c": 0.0,
#     "std_c": 1,
#     "time": 20,
# }

n_max = 4096
n_test = 240
n_val = 120
start = n_max - n_test - n_val
length = n_val + n_test

# for label in ["260", "489", "754"]:
for label in ["829"]:
    which_example = "piezo_conductivity"
    # label = "260"
    variant = "2"
    model, loader_dict = load_model(
        folder=f"../TrainedModels/Time_CNO_{which_example}_{variant}",
        which_example=which_example,
        steps=11,
        in_dim=3,
        out_dim=2,
        label=label,
    )
    end_time = constants["time"]

    best_mae = 1e6
    best_samples = []

    offset = 0
    times = [
        (
            torch.ones(1, 128, 128, requires_grad=False).type(torch.float32)
            * (t)
            / end_time,
            t,
        )
        for t in range(offset, end_time - offset)
    ]

    loss = torch.nn.L1Loss()
    for sample in range(start, start + length):
        if (sample - start) % 10 == 0:
            print(sample, best_mae, best_samples[-3:])
        mae = 0
        start_time = 0

        inputs_function = torch.tensor(
            solution[sample, start_time], requires_grad=False
        ).reshape(1, 128, 128)
        inputs_function = (inputs_function - constants["mean"]) / constants["std"]

        inputs_condition = torch.tensor(c[sample], requires_grad=False).reshape(
            1, 128, 128
        )

        for t, t_idx in times:
            time = (t_idx - start_time) / end_time
            inputs = torch.cat([inputs_function, inputs_condition, t], dim=0).unsqueeze(
                dim=0
            )

            labels = torch.tensor(solution[sample, t_idx], requires_grad=False).reshape(
                1, 128, 128
            )
            labels = (labels - constants["mean"]) / constants["std"]
            labels = torch.cat([labels, inputs_condition], dim=0)
            labels = labels.unsqueeze(dim=0)

            predicted = model.forward(
                inputs, torch.tensor(time, requires_grad=False)
            ).detach()

            all_loss = loss(predicted, labels)
            mae += all_loss
            # if mae > 1.5 * best_mae:
            #     break

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
