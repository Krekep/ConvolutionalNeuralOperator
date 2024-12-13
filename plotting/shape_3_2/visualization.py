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


def create_frames(model, dataset, constants, gif_name="cno_gauss.gif"):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

    end_time = constants["time"]
    sample_num = random.randint(0, len(dataset["solution"]) - 1)

    # for colorbar
    im_upper = ax[0, 0].imshow(torch.ones(128, 128))
    im_median = ax[1, 0].imshow(torch.ones(128, 128))
    im_lower_time = ax[2, 0].imshow(torch.ones(128, 128), vmin=0, vmax=1)
    im_lower_sol = ax[2, 1].imshow(torch.zeros(128, 128))
    im_lower_cond = ax[2, 2].imshow(torch.zeros(128, 128))
    cbar_upper = fig.colorbar(im_upper, ax=ax[0])
    cbar_median = fig.colorbar(im_median, ax=ax[1])
    cbar_lower_time = fig.colorbar(im_lower_time, ax=ax[2, 0])
    cbar_lower_diff_sol = fig.colorbar(im_lower_sol, ax=ax[2, 1])
    cbar_lower_diff_cond = fig.colorbar(im_lower_cond, ax=ax[2, 2])

    time_pairs = []
    for s_t in range(0, end_time):
        for c_t in range(s_t, end_time):
            time_pairs.append((s_t, c_t))

    def create_frame(time_index):
        start_time, t = time_pairs[time_index]
        time = (t - start_time) / end_time
        time_delta = torch.ones(1, 128, 128).type(torch.float32) * time

        inputs_function = torch.tensor(
            dataset["solution"][sample_num, start_time]
        ).reshape(1, 128, 128)
        inputs_condition = torch.tensor(dataset["c"][sample_num]).reshape(1, 128, 128)

        inputs_function = (inputs_function - constants["mean"]) / constants["std"]
        inputs_condition = (inputs_condition - constants["mean_c"]) / constants["std_c"]
        inputs = torch.cat(
            [inputs_function, inputs_condition, time_delta], dim=0
        ).unsqueeze(dim=0)

        labels = torch.tensor(dataset["solution"][sample_num, t]).reshape(1, 128, 128)
        labels = (labels - constants["mean"]) / constants["std"]
        labels = torch.cat([labels, inputs_condition], dim=0).unsqueeze(dim=0)

        predicted = model.forward(inputs, torch.tensor(t)).detach()
        difference_cond = labels[0][1] - predicted[0][1]
        difference_sol = labels[0][0] - predicted[0][0]
        all_loss = torch.nn.L1Loss()(predicted, labels)
        func_loss = torch.nn.L1Loss()(predicted[0][0], labels[0][0])
        ic_loss = torch.nn.L1Loss()(predicted[0][1], labels[0][1])

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
        med_min = min(
            np.min(inputs[0][1].numpy()),
            np.min(predicted[0][1].numpy()),
            np.min(labels[0][1].numpy()),
        )
        med_max = max(
            np.max(inputs[0][1].numpy()),
            np.max(predicted[0][1].numpy()),
            np.max(labels[0][1].numpy()),
        )
        diff_sol_max = np.max(difference_sol.numpy())
        diff_sol_min = np.min(difference_sol.numpy())
        diff_cond_max = np.max(difference_cond.numpy())
        diff_cond_min = np.min(difference_cond.numpy())

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

        ax[2, 1].imshow(difference_sol)
        ax[2, 1].set_title(f"Difference between solutions")

        ax[2, 2].imshow(difference_cond)
        ax[2, 2].set_title(f"Difference between conditions")

        im_upper.set_clim(up_min, up_max)
        im_median.set_clim(med_min, med_max)
        im_lower_sol.set_clim(diff_sol_min, diff_sol_max)
        im_lower_cond.set_clim(diff_cond_min, diff_cond_max)
        fig.suptitle(
            f"All MAE lose = {all_loss:.3f}, function loss = {func_loss:.3f}, initial condition loss = {ic_loss:.3f}"
        )
        return ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1], ax[0, 2], ax[1, 2]

    gif = animation.FuncAnimation(fig, create_frame, frames=len(time_pairs) // 2)
    gif.save(gif_name, dpi=300, writer="pillow", fps=2)


which_example = "wave_gauss"
label = "2"
variant = "3"
cno, loader_dict = load_model(
    folder=f"../../TrainedModels/Time_CNO_{which_example}_{variant}",
    which_example=which_example,
    steps=10 if which_example == "piezo_conductivity" else 7,
    in_dim=3,
    out_dim=2,
    label=label,
)
dataset_nc = netCDF4.Dataset("../../nc_data/res/Wave-Gauss.nc")
# dataset_nc = netCDF4.Dataset("../../gp_data/piezo_conductivity.nc")
solution = dataset_nc["solution"]
c = dataset_nc["c"]

# std_sol = np.std(solution).data.item()
# std_c = np.std(c).data.item()
# constants = {
#     "mean": np.mean(solution).data.item(),
#     "std": std_sol if not math.isclose(std_sol, 0) else 1,
#     "mean_c": np.mean(c).data.item(),
#     "std_c": std_c if not math.isclose(std_c, 0) else 1,
#     "time": 20,
# }

# wave gauss
constants = {
    "mean": 0.0334376316,
    "std": 0.1171879068,
    "mean_c": 2618.4593933,
    "std_c": 601.51658913,
    "time": 15,
}
create_frames(
    cno, dataset_nc, constants, gif_name=f"cno_{which_example}_{variant}_{label}.gif"
)
