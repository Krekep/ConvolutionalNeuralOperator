import random
import netCDF4
import torch
from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_frames(model, dataset, gif_name="cno_gauss.gif"):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    start_time = 0
    end_time = 15
    sample_num = random.randint(0, 10000)
    inputs_function = torch.tensor(dataset["solution"][sample_num, start_time]).reshape(
        1, 128, 128
    )
    inputs_condition = torch.tensor(dataset["c"][sample_num]).reshape(1, 128, 128)

    def create_frame(t):
        time = (t - start_time) / end_time
        time_delta = torch.ones(1, 128, 128).type(torch.float32) * time
        inputs = torch.cat(
            [inputs_function, inputs_condition, time_delta], dim=0
        ).unsqueeze(dim=0)

        labels = torch.tensor(dataset["solution"][sample_num, t]).reshape(1, 128, 128)
        labels = torch.cat([labels, inputs_condition], dim=0).unsqueeze(dim=0)

        predicted = model.forward(inputs, torch.tensor(t)).detach()
        all_loss = torch.nn.L1Loss()(predicted, labels)
        func_loss = torch.nn.L1Loss()(predicted[0][0], labels[0][0])
        ic_loss = torch.nn.L1Loss()(predicted[0][1], labels[0][1])

        plt.cla()
        ax[0, 0].imshow(inputs[0][0])
        ax[0, 0].set_title(f"Input at time {start_time}")
        ax[1, 0].imshow(inputs_condition[0])
        ax[1, 0].set_title(f"Initial conditions")

        ax[0, 1].imshow(predicted[0][0])
        ax[0, 1].set_title(f"Predicted at time {t}")
        ax[1, 1].imshow(predicted[0][1])
        ax[1, 1].set_title(f"Predicted conditions")

        ax[0, 2].imshow(labels[0][0])
        ax[0, 2].set_title(f"True at time {t}")
        ax[1, 2].imshow(labels[0][1])
        ax[1, 2].set_title(f"Initial conditions")

        fig.suptitle(
            f"All lose = {all_loss:.3f}, function loss = {func_loss:.3f}, initial condition loss = {ic_loss:.3f}"
        )
        return ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1], ax[0, 2], ax[1, 2]

    gif = animation.FuncAnimation(fig, create_frame, frames=15)
    gif.save(gif_name, dpi=300, writer="pillow", fps=3)


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
print(wave_gauss_nc)
print(wave_gauss_ic_nc)

create_frames(cno, wave_gauss_nc)
