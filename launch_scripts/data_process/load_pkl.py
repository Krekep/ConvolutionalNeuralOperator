import pickle

import h5py
import netCDF4
import numpy as np

# sample, times, num_points, (x, y, v)

with open("reservoir_pressure_unstruct.pkl", "rb") as f:
    data: np.ndarray = pickle.load(f)
print(data.shape)
print(type(data))
c_data = data[:, 0, :, :]
sol_data = data[:, 1:, :, :]


new_nc = netCDF4.Dataset("piezo_conductivity.nc", mode="w")

sample_dim = new_nc.createDimension("sample", 990)
time_dim = new_nc.createDimension("time", 11)
points_dim = new_nc.createDimension("points", 5235)
data_dim = new_nc.createDimension("data", 3)

solution = new_nc.createVariable(
    "solution", np.float32, dimensions=[sample_dim, time_dim, points_dim, data_dim]
)
solution[:] = sol_data

# c_data = np.zeros((4096, 128, 128))
c = new_nc.createVariable(
    "c", np.float32, dimensions=[sample_dim, points_dim, data_dim]
)
c[:] = c_data

new_nc.close()

max_x = np.max(data[:, :, :, 0])
min_x = np.min(data[:, :, :, 0])
max_y = np.max(data[:, :, :, 1])
min_y = np.min(data[:, :, :, 1])
for sample in range(990):
    for t1 in range(11):
        if not (data[sample, t1, :, :3] == data[sample, t1 + 1, :, :3]).all():
            print(sample, t1, t1 + 1)

print(max_x, max_y, min_x, min_y)

# temp = h5py.File("piezo_conductivity.nc", "r")
# # print(temp["solution"][2][3])
# # print(new_data[2][3])
#
# solution = temp["solution"]
# c = temp["c"]
#
# m = np.mean(solution)
# md = m.data
#
# constants = {
#     "mean": np.mean(solution),
#     "std": np.std(solution),
#     "mean_c": np.mean(c),
#     "std_c": np.std(c),
#     "time": 20,
# }
#
# print(constants)
