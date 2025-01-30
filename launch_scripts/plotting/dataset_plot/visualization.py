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


def create_frames(dataset, sample_num=None, gif_name="cno_gauss.gif"):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

    sample_num = (
        random.randint(0, len(dataset["solution"]) - 1)
        if sample_num is None
        else sample_num
    )

    def create_frame(time_index):
        plt.cla()
        ax[0].imshow(dataset["solution"][sample_num, time_index])
        ax[0].set_title(f"Input at time {time_index}")

        ax[1].imshow(dataset["c"][sample_num])
        ax[1].set_title(f"Wells at time {time_index}")

        return ax[0], ax[1]

    gif = animation.FuncAnimation(fig, create_frame, frames=20)
    gif.save(gif_name, dpi=300, writer="pillow", fps=2)


path_to_folder = ""
dataset_nc = netCDF4.Dataset(f"{path_to_folder}/piezo_conductivity.nc")
solution = dataset_nc["solution"]
c = dataset_nc["c"]

for sample_num in [0, 3938, 3925, 3883, 3832, 3935, 4095, 4050, 3975, 4067]:
    # sample_num = 587
    create_frames(
        dataset_nc, sample_num=sample_num, gif_name=f"test_sample_{sample_num}.gif"
    )
