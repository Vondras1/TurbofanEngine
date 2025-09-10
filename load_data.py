import numpy as np

# path
file_path = "NASA-Turbofan-data/data/train_FD001.txt"

# Load data as a matrix (number of cycles x 26)
data = np.loadtxt(file_path)

print(data.shape)
print(data[:5, :])