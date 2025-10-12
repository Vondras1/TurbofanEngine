# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display

# %% Data loading -----------------------------------------------------------------------------------------------
def filter_no_variance(data):
    # Select only sensor columns
    sensor_cols = [col for col in data.columns if "sensor measurement" in col]

    # # Keep only sensors with non-zero variance and with their orig index
    valid_cols = []
    filtered_cols = []
    for i, col in enumerate(sensor_cols):
        if data[col].var() > 1e-8:
            valid_cols.append((i, col))
        else:
            filtered_cols.append(col)

    print("Zero variance columns (constant values):", filtered_cols)

    data = data.drop(columns=filtered_cols)
    return data, filtered_cols

def add_mean_max_min_column(data, window = 25):
  sensor_cols = [c for c in data.columns if "sensor measurement" in c]
  for i, col in enumerate(sensor_cols):
      # SM1_mean is mean of window values from sensor measurement 1
      data[f"SM{i+1}_mean" ] = data.groupby("unit number")[col].transform(
          lambda x: x.rolling(window=window, min_periods=1).mean()
      )
      data[f"SM{i+1}_max"] = data.groupby("unit number")[col].transform(
          lambda x: x.rolling(window=window, min_periods=1).max()
      )
      data[f"SM{i+1}_min"] = data.groupby("unit number")[col].transform(
          lambda x: x.rolling(window=window, min_periods=1).min()
      )

def from_file(train_file_name):
    names = [
      "unit number",
      "time, in cycles",
      "operational setting 1",
      "operational setting 2",
      "operational setting 3"
        ]

    sen_measurements = [f"sensor measurement {i}" for i in range(1, 22)]
    names = names + sen_measurements
    data = pd.read_csv(train_file_name, sep=r'\s+', names=names)

    return data


def train_from_file(file_name, validation_partition = 0.2):
    data = from_file(file_name)

    data, filtered_cols = filter_no_variance(data)
    add_mean_max_min_column(data)

    #COUNTING RUL
    units = dict(tuple(data.groupby("unit number")))
    new_units = []
    for u_id in units:
        u = units[u_id]
        u = u.drop(columns=[
                        "operational setting 1",
                        "operational setting 2",
                        "operational setting 3"])

        failure_time = u.shape[0] + 1
        u["RUL"] = failure_time - u["time, in cycles"]
        new_units.append(u)

    units = new_units

    # SPLITTING VALIDATION / TRAIN
    validation_unit_count = math.ceil(len(units) * validation_partition)
    np.random.seed(42)
    np.random.shuffle(units)

    validation_units = units[:validation_unit_count]
    train_units = units[validation_unit_count:]

    validation_data = pd.concat(validation_units)
    train_data = pd.concat(train_units)

    return train_data, validation_data, filtered_cols

def test_from_file(file_name, gt_file_name, dropped_cols):
    data = from_file(file_name)
    data = data.drop(columns=dropped_cols)

    add_mean_max_min_column(data)

    #read file, where there is only one number on a line into a vector of these numbers
    gt = np.loadtxt(gt_file_name)

    #COUNTING RUL
    units = dict(tuple(data.groupby("unit number")))
    new_units = []
    i = 0
    for u_id in units:
        u = units[u_id]
        u = u.drop(columns=[
                        "operational setting 1",
                        "operational setting 2",
                        "operational setting 3"])
        RUL = gt[i]
        u["RUL"] = int(RUL + u.shape[0] + 1) - u["time, in cycles"]
        i += 1
        new_units.append(u)

    units = new_units

    test_data = pd.concat(units)
    return test_data


train_data1, validation_data1, dropped_cols1 = train_from_file("./NASA-Turbofan-data/data/train_FD001.txt")
test_data1 = test_from_file("./NASA-Turbofan-data/data/test_FD001.txt", "./NASA-Turbofan-data/data/RUL_FD001.txt", dropped_cols=dropped_cols1)

train_data2, validation_data2, dropped_cols2 = train_from_file("./NASA-Turbofan-data/data/train_FD002.txt")
test_data2 = test_from_file("./NASA-Turbofan-data/data/test_FD002.txt", "./NASA-Turbofan-data/data/RUL_FD002.txt", dropped_cols=dropped_cols2)

train_data3, validation_data3, dropped_cols3 = train_from_file("./NASA-Turbofan-data/data/train_FD003.txt")
test_data3 = test_from_file("./NASA-Turbofan-data/data/test_FD003.txt", "./NASA-Turbofan-data/data/RUL_FD003.txt", dropped_cols=dropped_cols3)

train_data4, validation_data4, dropped_cols4 = train_from_file("./NASA-Turbofan-data/data/train_FD004.txt")
test_data4 = test_from_file("./NASA-Turbofan-data/data/test_FD004.txt", "./NASA-Turbofan-data/data/RUL_FD004.txt", dropped_cols=dropped_cols4)

# %% Data normalization -------------------------------------------------------------------------------------------
def normalizeRUL(Y, y_min, y_max):
    Y_scaled = (Y - y_min) / (y_max - y_min)
    return Y_scaled

def normalize_data(train_data, validation_data, test_data, normRUL=False):
    sensor_cols = [
        c for c in train_data.columns 
        if (("sensor measurement" in c) or ("mean" in c) or ("max" in c) or ("min" in c))
        and c not in ["unit number", "time_in_cycles"]
    ]
    print(sensor_cols)

    mu_train = train_data[sensor_cols].mean(axis=0)
    sd_train = train_data[sensor_cols].std(axis=0, ddof=0)

    sd_r = sd_train.replace(0, 1)  # prevence dělení nulou

    train_data[sensor_cols]      = (train_data[sensor_cols     ] - mu_train) / sd_r
    validation_data[sensor_cols] = (validation_data[sensor_cols] - mu_train) / sd_r
    test_data[sensor_cols]       = (test_data[sensor_cols      ] - mu_train) / sd_r

    #Normalize RUL
    if normRUL==True:
        y_min, y_max = train_data["RUL"].min(), train_data["RUL"].max()
        train_data["RUL"] = normalizeRUL(train_data["RUL"], y_min, y_max)
        validation_data["RUL"] = normalizeRUL(validation_data["RUL"], y_min, y_max)
        test_data["RUL"] = normalizeRUL(test_data["RUL"], y_min, y_max)

    return train_data, validation_data, test_data, mu_train, sd_train

# normalize
train_data1, validation_data1, test_data1, mu_train, sd_train = normalize_data(train_data1, validation_data1, test_data1)
display(test_data1)



# %% KLS starts here -----------------------------------------------------------------------------------------------
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% KPLS

X_train = train_data1.iloc[:, 2:-1]
Y_train = train_data1.iloc[:, -1]

X_validation = validation_data1.iloc[:, 2:-1]
Y_validation = validation_data1.iloc[:, -1]


# %% find gamma and latent variables

gamma_list = [0.0005, 0.001, 0.0025, 0.05, 0.025, 0.01]
latent_vars_list = list(range(1, 11))  # 1..10

# --- Store results ---
Q2_matrix = np.full((len(gamma_list), len(latent_vars_list)), np.nan)

# --- Loop over gamma and latent variable counts ---
for gi, gamma in enumerate(gamma_list):
    # Compute kernel & center it
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    centerer = KernelCenterer().fit(K_train)
    K_train_c = centerer.transform(K_train)

    K_validation = rbf_kernel(X_validation, X_train, gamma=gamma)
    K_validation_c = centerer.transform(K_validation)

    # Loop over number of latent variables
    for li, n_components in enumerate(latent_vars_list):
        pls = PLSRegression(n_components=n_components)
        pls.fit(K_train_c, Y_train)

        # Predict on validation kernel
        Y_pred_val = pls.predict(K_validation_c).ravel()

        # Compute Q²
        PRESS = np.sum((Y_validation - Y_pred_val)**2)
        TSS = np.sum((Y_validation - np.mean(Y_train))**2)
        Q2 = 1 - PRESS / TSS if TSS > 0 else np.nan

        Q2_matrix[gi, li] = Q2

# --- Find best gamma and LV ---
best_idx = np.unravel_index(np.nanargmax(Q2_matrix), Q2_matrix.shape)
best_gamma = gamma_list[best_idx[0]]
best_lv = latent_vars_list[best_idx[1]]
best_Q2 = Q2_matrix[best_idx]

print("Best parameters:")
print(f"  γ (gamma) = {best_gamma}")
print(f"  Latent variables = {best_lv}")
print(f"  Q² (validation) = {best_Q2:.4f}")

# --- Optional: Plot heatmap of Q² ---
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(Q2_matrix, annot=True, fmt=".2f", xticklabels=latent_vars_list, yticklabels=gamma_list, cmap='viridis')
plt.xlabel('# Latent Variables')
plt.ylabel('Gamma')
plt.title('Validation Q² for different (gamma, LV) combinations')
plt.show()

# %% Fitting the model
gamma = 0.01
n_component = 5

K_train = rbf_kernel(X_train, X_train, gamma=gamma)   # symmetric kernel Gram matrix
centerer = KernelCenterer().fit(K_train)
K_train_c = centerer.transform(K_train)

# 3. Fit PLS regression on the kernel matrix
pls = PLSRegression(n_components)  # e.g., 10 latent components
pls.fit(K_train_c, Y_train)

# 5. Predict and evaluate
Y_pred = pls.predict(K_validation_c)

rmse = np.sqrt(mean_squared_error(Y_validation, Y_pred))
mae  = mean_absolute_error(Y_validation, Y_pred)
print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# %% Evaluate the test dataset

X_test = test_data1.iloc[:, 2:-1]
Y_test = test_data1.iloc[:, -1]

K_test = rbf_kernel(X_test, X_train, gamma) 
K_test_c = centerer.transform(K_test)
Y_pred = pls.predict(K_test_c)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae  = mean_absolute_error(Y_test, Y_pred)
print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}")