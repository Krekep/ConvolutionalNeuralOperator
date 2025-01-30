import math
import random
import netCDF4
import numpy as np
import torch
from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_frames(
    model,
    dataset,
    constants,
    main_start_time=0,
    sample_num=None,
    gif_name="cno_gauss.gif",
):
    """Function for create model prediction gif"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.delaxes(ax[1, 2])

    end_time = constants["time"]
    sample_num = (
        random.randint(0, len(dataset["solution"]) - 1)
        if sample_num is None
        else sample_num
    )

    # for colorbar
    im_upper = ax[0, 0].imshow(torch.ones(128, 128))
    im_lower_time = ax[1, 0].imshow(torch.ones(128, 128), vmin=0, vmax=1)
    im_lower_sol = ax[1, 1].imshow(torch.zeros(128, 128))
    im_lower_cond = ax[1, 2].imshow(torch.zeros(128, 128))
    cbar_upper = fig.colorbar(im_upper, ax=ax[0])
    cbar_lower_time = fig.colorbar(im_lower_time, ax=ax[1, 0])
    cbar_lower_diff_sol = fig.colorbar(im_lower_sol, ax=ax[1, 1])

    time_pairs = []
    for s_t in range(main_start_time, end_time):
        for c_t in range(s_t, end_time):
            time_pairs.append((s_t, c_t))

    def create_frame(time_index):
        start_time, t = time_pairs[time_index]
        time = (t - start_time) / end_time
        time_delta = torch.ones(1, 128, 128).type(torch.float32) * time

        inputs_function = torch.tensor(
            dataset["solution"][sample_num, start_time]
        ).reshape(1, 128, 128)

        inputs_function = (inputs_function - constants["mean"]) / constants["std"]
        inputs = torch.cat([inputs_function, time_delta], dim=0).unsqueeze(dim=0)

        labels = torch.tensor(dataset["solution"][sample_num, t]).reshape(1, 128, 128)
        labels = (labels - constants["mean"]) / constants["std"]
        labels = labels.unsqueeze(dim=0)

        # predicted = model.forward(inputs, torch.tensor(t)).detach()
        predicted = model.forward(inputs, torch.tensor(time)).detach()
        difference_sol = labels[0][0] - predicted[0][0]
        all_loss = torch.nn.L1Loss()(predicted, labels)
        func_loss = torch.nn.L1Loss()(predicted[0][0], labels[0][0])

        up_min = min(
            np.min(inputs[0][0].numpy()),
            np.min(predicted[0][0].numpy()),
            np.min(labels[0][0].numpy()),
        )
        up_max = max(
            np.max(inputs[0][0].numpy()),
            np.max(predicted[0][0].numpy()),
            np.max(labels[0][0].numpy()),
        )
        diff_sol_max = np.max(difference_sol.numpy())
        diff_sol_min = np.min(difference_sol.numpy())

        plt.cla()
        ax[0, 0].imshow(inputs[0][0], vmin=up_min, vmax=up_max)
        ax[0, 0].set_title(f"Input at time {start_time}")

        ax[0, 1].imshow(predicted[0][0], vmin=up_min, vmax=up_max)
        ax[0, 1].set_title(f"Predicted at time {t}")

        ax[0, 2].imshow(labels[0][0], vmin=up_min, vmax=up_max)
        ax[0, 2].set_title(f"True at time {t}")

        ax[1, 0].imshow(time_delta[0], vmin=0, vmax=1)
        ax[1, 0].set_title(f"Input time delta {time}")

        ax[1, 1].imshow(difference_sol)
        ax[1, 1].set_title(f"Difference between solutions")

        im_upper.set_clim(up_min, up_max)
        im_lower_sol.set_clim(diff_sol_min, diff_sol_max)
        fig.suptitle(f"All MAE lose = {all_loss:.3f}, function loss = {func_loss:.3f}")
        return ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1], ax[0, 2], ax[1, 2]

    gif = animation.FuncAnimation(fig, create_frame, frames=len(time_pairs) // 4)
    gif.save(gif_name, dpi=300, writer="pillow", fps=2)


# input is (function, time_delta), output is (function)
which_example = "piezo_conductivity_no_condition"
label = "440"  # model label
variant = "2"
cno, loader_dict = load_model(
    folder=f"../../TrainedModels/Time_CNO_{which_example}_{variant}",
    which_example=which_example,
    steps=11,
    in_dim=2,
    out_dim=1,
    label=label,
)
path_to_folder = ""
dataset_nc = netCDF4.Dataset(f"{path_to_folder}/piezo_conductivity.nc")
solution = dataset_nc["solution"]
c = dataset_nc["c"]

constants = {
    "mean": -0.004365722648799419,
    "std": 0.7487624287605286,
    "time": 20,
}


n_max = 4096
n_test = 240
n_val = 120
start = n_max - n_test - n_val
length = n_val

for sample_num in [
    4034,
    4018,
    3989,
    3949,
    3904,
    3903,
    3891,
    4049,
]:  # samples for which we build the gif
    # sample_num = 587
    create_frames(
        cno,
        dataset_nc,
        constants,
        main_start_time=0,
        sample_num=sample_num,
        gif_name=f"cno_{which_example}_{variant}_{label}_sample_{sample_num}.gif",
    )
