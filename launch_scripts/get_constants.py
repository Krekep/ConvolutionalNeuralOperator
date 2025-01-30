import math

import netCDF4
import numpy as np

path_to_folder = ""
dataset_nc = netCDF4.Dataset("/piezo_conductivity.nc")
print(dataset_nc.variables)
print(dataset_nc.dimensions)

solution = dataset_nc["solution"]
c = dataset_nc["c"]

std_sol = np.std(solution).data.item()
std_c = np.std(c).data.item()
constants = {
    "mean": np.mean(solution).data.item(),
    "std": std_sol if not math.isclose(std_sol, 0) else 1,
    "mean_c": np.mean(c).data.item(),
    "std_c": std_c if not math.isclose(std_c, 0) else 1,
    "time": 20,
}
# {'mean': -0.004365722648799419, 'std': 0.7487624287605286, 'mean_c': -1.512402399497792e-12, 'std_c': 1.387237325012336e-09, 'time': 20}
#
print(constants)
