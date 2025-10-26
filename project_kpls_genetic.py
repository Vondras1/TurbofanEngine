#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import warnings

warnings.filterwarnings("ignore", message="y residual is constant at iteration")

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

# train_data2, validation_data2, dropped_cols2 = train_from_file("./NASA-Turbofan-data/data/train_FD002.txt")
# test_data2 = test_from_file("./NASA-Turbofan-data/data/test_FD002.txt", "./NASA-Turbofan-data/data/RUL_FD002.txt", dropped_cols=dropped_cols2)

# train_data3, validation_data3, dropped_cols3 = train_from_file("./NASA-Turbofan-data/data/train_FD003.txt")
# test_data3 = test_from_file("./NASA-Turbofan-data/data/test_FD003.txt", "./NASA-Turbofan-data/data/RUL_FD003.txt", dropped_cols=dropped_cols3)

# train_data4, validation_data4, dropped_cols4 = train_from_file("./NASA-Turbofan-data/data/train_FD004.txt")
# test_data4 = test_from_file("./NASA-Turbofan-data/data/test_FD004.txt", "./NASA-Turbofan-data/data/RUL_FD004.txt", dropped_cols=dropped_cols4)

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

# %% 
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from genetic import run_genetic

# %% KPLS starts here

def fit_kpls(train_data, validation_data, test_data, params = None):
    # used for training
    X_train = train_data.iloc[:, 2:-1]
    Y_train = train_data.iloc[:, -1]

    if params == None:
        gamma, n_components, q2, q2_log = run_genetic(
            population_size = 30,
            generations     = 25,
            data=validation_data,
        )

        # Split into two columns
        iters = q2_log[:, 0]
        Q2s = q2_log[:, 1]

        # Plot
        plt.figure(figsize=(6,4))
        plt.plot(iters, Q2s, marker='o', linestyle='-', color='b')
        plt.xlabel("Iteration")
        plt.ylabel("Q2")
        plt.title("Q2 vs Iteration")
        plt.grid(True)
        plt.show()
    else:
        gamma = params[0]
        n_components = params[1]

    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    centerer = KernelCenterer()
    K_train_c = centerer.fit_transform(K_train)
    
    pls = PLSRegression(n_components)
    pls.fit(K_train_c, Y_train)
    
    # evaluating test partition
    X_test = test_data1.iloc[:, 2:-1]
    Y_test = test_data1.iloc[:, -1]
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    K_test_c = centerer.transform(K_test)

    Y_pred = pls.predict(K_test_c)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae  = mean_absolute_error(Y_test, Y_pred)
    print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    test_unit = test_data[test_data["unit number"] == 20]
    # Prepare inputs
    Xunit = test_unit.iloc[:, 2:-1].to_numpy()
    K_unit = rbf_kernel(Xunit, X_train, gamma=gamma)
    Kunit_c = centerer.transform(K_unit)
    Yunit = test_unit["RUL"].to_numpy()

    # Predict
    Ypred = pls.predict(Kunit_c).ravel()
    Ypred = np.clip(Ypred, 0, None)

    # Observed vs Predicted (scatter)
    plt.figure(figsize=(5, 5))
    plt.scatter(Yunit, Ypred, s=15, c="tab:blue", alpha=0.7)
    plt.plot([Yunit.min(), Yunit.max()], [Yunit.min(), Yunit.max()], "k--", lw=1.2)
    plt.xlabel("Observed RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Validation - Observed vs Predicted (Unit {20})")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

params = None
#params = [ 0.0025, 5]
fit_kpls(train_data1, validation_data1, test_data1, params)



