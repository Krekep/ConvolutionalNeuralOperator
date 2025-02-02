import netCDF4
import numpy as np

with open("dataset_navier_stokes_64x64.npy", "rb") as f:
    data: np.ndarray = np.load(f)
print(data.shape)
print(type(data))
sol_data = data[:, :, :, :]

new_nc = netCDF4.Dataset("navier_stokes_no_condition.nc", mode="w")

sample_dim = new_nc.createDimension("sample", 1000)
time_dim = new_nc.createDimension("time", 20)
x_dim = new_nc.createDimension("x", 64)
y_dim = new_nc.createDimension("y", 64)

solution = new_nc.createVariable(
    "solution", np.float32, dimensions=[sample_dim, time_dim, x_dim, y_dim]
)
solution[:] = sol_data

new_nc.close()
