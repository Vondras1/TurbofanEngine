import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def split_into_subsets(data, unique_ids):
    # Split the dataset into subsets by unit ID
    Mset = []
    for ui in unique_ids:
        one_cycle_M = data[data[:,0]==ui, :]
        Mset.append(one_cycle_M)
    return Mset

def check_data_integrity():
    #TODO
    # missing values
    # extreme values
    # ...
    pass

def visualize():
    # TODO
    # Jak to udelame s vizualizacem?
    pass

# path
file_path = "NASA-Turbofan-data/data/train_FD001.txt"

# Load data as a matrix (number of cycles x 26)
# Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
# 1)	unit number
# 2)	time, in cycles
# 3)	operational setting 1
# 4)	operational setting 2
# 5)	operational setting 3
# 6)	sensor measurement  1
# 7)	sensor measurement  2
# ...
# 26)	sensor measurement  21
data = np.loadtxt(file_path)
print(f"{data.shape=}")

# Extract all unique unit IDs
unique_ids = np.unique(data[:, 0].astype(int)) 

# Split the dataset into subsets by unit ID
Mset = split_into_subsets(data, unique_ids)

check_data_integrity() # TODO

visualize() # TODO

## PCA 
X = data[:, 5:] # Consider only senory data
print(X.shape)

Xmean = X.mean(axis=0) # mean
Xstd = X.std(axis=0) # std

X_std_filtered = np.where(Xstd == 0, 1.0, Xstd)
Xnorm = (X - Xmean) / X_std_filtered # normalized

pca = PCA()
Xpca = pca.fit_transform(Xnorm)

# expaleined variation
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("PCs")
plt.ylabel("Cumulative explained variance")
plt.show()