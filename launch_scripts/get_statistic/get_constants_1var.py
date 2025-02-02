import math

import netCDF4
import numpy as np

path_to_folder = ""
dataset_nc = netCDF4.Dataset("../data_process/navier_stokes_no_condition.nc")
print(dataset_nc.variables)
print(dataset_nc.dimensions)

solution = dataset_nc["solution"]

std_sol = np.std(solution).data.item()
constants = {
    "mean": np.mean(solution).data.item(),
    "std": std_sol if not math.isclose(std_sol, 0) else 1,
    "time": 20,
}
# {'mean': 1.929536892930628e-06, 'std': 1.131550669670105, 'time': 20}
#
print(constants)
