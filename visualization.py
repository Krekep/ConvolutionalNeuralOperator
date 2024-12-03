import random
import netCDF4
import numpy as np
import torch
from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_frames(model, dataset, constants, gif_name="cno_gauss.gif"):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.delaxes(ax[2, 1])
    fig.delaxes(ax[2, 2])

    start_time = 0
    end_time = constants["time"]
    sample_num = random.randint(0, 10000)
    inputs_function = torch.tensor(dataset["solution"][sample_num, start_time]).reshape(
        1, 128, 128
    )
    inputs_condition = torch.tensor(dataset["c"][sample_num]).reshape(1, 128, 128)

    inputs_function = (inputs_function - constants["mean"]) / constants["std"]
    inputs_condition = (inputs_condition - constants["mean_c"]) / constants["std_c"]

    # for colorbar
    im_upper = ax[0, 0].imshow(inputs_function[0])
    im_median = ax[1, 0].imshow(inputs_condition[0])
    im_lower = ax[2, 0].imshow(torch.ones(128, 128), vmin=0, vmax=1)
    cbar_upper = fig.colorbar(im_upper, ax=ax[0])
    cbar_median = fig.colorbar(im_median, ax=ax[1])
    cbar_lower = fig.colorbar(im_lower, ax=ax[2, 0])

    def create_frame(t):
        time = (t - start_time) / end_time
        time_delta = torch.ones(1, 128, 128).type(torch.float32) * time
        inputs = torch.cat(
            [inputs_function, inputs_condition, time_delta], dim=0
        ).unsqueeze(dim=0)

        labels = torch.tensor(dataset["solution"][sample_num, t]).reshape(1, 128, 128)
        labels = (labels - constants["mean"]) / constants["std"]
        labels = torch.cat([labels, inputs_condition], dim=0).unsqueeze(dim=0)

        predicted = model.forward(inputs, torch.tensor(t)).detach()
        all_loss = torch.nn.L1Loss()(predicted, labels)
        func_loss = torch.nn.L1Loss()(predicted[0][0], labels[0][0])
        ic_loss = torch.nn.L1Loss()(predicted[0][1], labels[0][1])

        up_min = min(np.min(inputs[0][0].numpy()), np.min(predicted[0][0].numpy()), np.min(labels[0][0].numpy()))
        up_max = max(np.max(inputs[0][0].numpy()), np.max(predicted[0][0].numpy()), np.max(labels[0][0].numpy()))
        med_min = min(np.min(inputs[0][1].numpy()), np.min(predicted[0][1].numpy()), np.min(labels[0][1].numpy()))
        med_max = max(np.max(inputs[0][1].numpy()), np.max(predicted[0][1].numpy()), np.max(labels[0][1].numpy()))

        plt.cla()
        ax[0, 0].imshow(inputs[0][0], vmin=up_min, vmax=up_max)
        ax[0, 0].set_title(f"Input at time {start_time}")
        ax[1, 0].imshow(inputs_condition[0], vmin=med_min, vmax=med_max)
        ax[1, 0].set_title(f"Input initial conditions")

        ax[0, 1].imshow(predicted[0][0], vmin=up_min, vmax=up_max)
        ax[0, 1].set_title(f"Predicted at time {t}")
        ax[1, 1].imshow(predicted[0][1], vmin=med_min, vmax=med_max)
        ax[1, 1].set_title(f"Predicted conditions")

        ax[0, 2].imshow(labels[0][0], vmin=up_min, vmax=up_max)
        ax[0, 2].set_title(f"True at time {t}")
        ax[1, 2].imshow(labels[0][1], vmin=med_min, vmax=med_max)
        ax[1, 2].set_title(f"True initial conditions")

        ax[2, 0].imshow(time_delta[0], vmin=0, vmax=1)
        ax[2, 0].set_title(f"Input time delta {time}")

        im_upper.set_clim(up_min, up_max)
        im_median.set_clim(med_min, med_max)
        fig.suptitle(
            f"All lose = {all_loss:.3f}, function loss = {func_loss:.3f}, initial condition loss = {ic_loss:.3f}"
        )
        return ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1], ax[0, 2], ax[1, 2]

    gif = animation.FuncAnimation(fig, create_frame, frames=end_time)
    gif.save(gif_name, dpi=300, writer="pillow", fps=2)


which_example = "wave_gauss"
cno, loader_dict = load_model(
    folder=f"TrainedModels/Time_CNO_{which_example}_1",
    which_example=which_example,
    in_dim=3,
    out_dim=2,
    label="794",
)
wave_gauss_nc = netCDF4.Dataset("nc_data/res/Wave-Gauss.nc")
wave_gauss_ic_nc = netCDF4.Dataset("nc_data/c_0.nc")

constants = {
    "mean": 0.0334376316,
    "std": 0.1171879068,
    "mean_c": 2618.4593933,
    "std_c": 601.51658913,
    "time": 15,
}
create_frames(cno, wave_gauss_nc, constants)
