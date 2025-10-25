#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = (
    "ignore:y residual is constant at iteration:UserWarning:sklearn.cross_decomposition._pls"
)

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"y residual is constant at iteration \d+",
    category=UserWarning,
    module=r"sklearn\.cross_decomposition\._pls"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

# %% 
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, chi2_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from optimizer import optimize
from kernel import cauchy_kernel, matern12_kernel

# %% KPLS starts here

def fit_model(X,y, n_lv, kernel_f, gamma):
    K_train = kernel_f(X, X, gamma=gamma)
    centerer = KernelCenterer()
    K_train_c = centerer.fit_transform(K_train)
    
    pls = PLSRegression(n_lv, scale=False)
    pls.fit(K_train_c, y)
    return pls, centerer

def count_q2(y_val, y_pred, y_train):
    ss_res = np.sum((y_val - y_pred.ravel())**2) 
    ss_tot = np.sum((y_val - np.mean(y_train))**2)
    return 1 - ss_res/ss_tot 

def predict(X_train, y_train, X, y, pls, centerer, gamma, kernel_f):
    K_validation   = kernel_f(X, X_train, gamma=gamma)
    K_validation_c = centerer.transform(K_validation)
    y_pred = pls.predict(K_validation_c).ravel()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    q2   = count_q2(y, y_pred, y_train)

    return rmse, mae, q2


def fit_kpls(train_data, validation_data, kernel_f=rbf_kernel, method_name="RBF_kernel"):

    output = []

    #optimize gamma on training data
    opt_results = optimize(train_data, max_lv=13, folds=10, kernel_f=kernel_f, method_name=method_name)
    best_row = opt_results.loc[opt_results["q2"].idxmax()]
    print(f"Best TRAIN n_lv={int(best_row.n_lv)}, gamma={best_row.gamma:.6g}, Q^2={best_row.q2:.4f}")
    
    #prepare datasets with only sensor data
    X_train = train_data.iloc[:, 2:-1]
    y_train = train_data.iloc[:, -1]

    X_validation = validation_data.iloc[:, 2:-1]
    y_validation = validation_data.iloc[:, -1]

    #eval validation dataset
    for _,res in opt_results.iterrows():
        gamma = res["gamma"]
        n_lv = int(res["n_lv"])

        train_q2   = res["q2"]
        pls, centerer = fit_model(X_train, y_train, n_lv, gamma)
        validation_rmse, validation_mae, validation_q2 = predict(
            X_train, y_train, X_validation, y_validation, pls, centerer, gamma=kernel_f)

        output.append({
            "n_lv": n_lv,
            "gamma": gamma,
            "q2_validation": validation_q2,
            "q2_train_cv": train_q2,
            "rmse_val": validation_rmse,
            "mae_val": validation_mae,
            "method_name": method_name,
            "kernel_f": kernel_f
        })

    return pd.DataFrame(output)

def eval_test(test_data, train_data, train_results, kernel_f, method_name):
    output = []

    #prepare datasets with only sensor data
    X_train = train_data.iloc[:, 2:-1]
    y_train = train_data.iloc[:, -1]

    X_test = test_data.iloc[:, 2:-1]
    y_test = test_data.iloc[:, -1]

    best_row_train = train_results.loc[train_results["q2_train_cv"].idxmax()]
    best_row_val = train_results.loc[train_results["q2_validation"].idxmax()]

    gamma_cv = best_row_train["gamma"]
    n_lv_cv  = int(best_row_train["n_lv"])

    gamma_val = best_row_val["gamma"]
    n_lv_val = int(best_row_val["n_lv"])

    # train the models
    pls_cv , centerer_cv  = fit_model(X_train, n_lv_cv , kernel_f, gamma_cv )
    pls_val, centerer_val = fit_model(X_train, n_lv_val, kernel_f, gamma_val)

    rmse_cv, mae_cv, q2_cv = predict(X_train, y_train, X_test, y_test, pls_cv, centerer_cv, gamma_cv, kernel_f)
    rmse_val, mae_val, q2_val = predict(X_train, y_train, X_test, y_test, pls_val, centerer_val, gamma_val, kernel_f)
    
    test_row_cv = best_row_train.copy()
    test_row_cv["rmse_test"] = rmse_cv
    test_row_cv["mae_test"] = mae_cv
    test_row_cv["q2_test"] = q2_cv

    test_row_val = best_row_val.copy()
    test_row_val["rmse_test"] = rmse_val
    test_row_val["mae_test"] = mae_val
    test_row_val["q2_test"] = q2_val

    # Combine into one output DataFrame
    output = pd.DataFrame([test_row_cv, test_row_val])
    return output

def plot_unit(train_data, test_data, params, dataset_name, unit_id):
    gamma = params.gamma
    n_lv = params.n_lv
    kernel_f = params.kernel_f

    X_train = train_data.iloc[:, 2:-1]
    y_train = train_data.iloc[:, -1]

    pls, centerer = fit_model(X_train, y_train, n_lv, kernel_f, gamma)
    test_unit = test_data[test_data["unit number"] == unit_id]

    Xunit = test_unit.iloc[:, 2:-1].to_numpy()
    K_unit = kernel_f(Xunit, X_train, gamma=gamma)
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
    plt.title(f"Validation - Observed vs Predicted (Unit {unit_id})")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def process_dataset(train_data, validation_data, test_data, dataset_name):
    print("RBF kernel")
    train_results_rbf = fit_kpls(train_data, validation_data, kernel_f=rbf_kernel, method_name="RBF")
    test_results_rbf = eval_test(test_data, train_data, train_results_rbf, kernel_f=rbf_kernel, method_name="RBF")

    print("Laplacian kernel")
    train_results_laplac = fit_kpls(train_data, validation_data, kernel_f=laplacian_kernel, method_name="Laplacian")
    test_results_laplac = eval_test(test_data, train_data, train_results_laplac, kernel_f=laplacian_kernel, method_name="Laplacian")

    # print("Matern12 kernel")
    # train_results_matern = fit_kpls(train_data, validation_data, kernel_f=matern12_kernel, method_name="Matern12")
    # test_results_matern = eval_test(test_data, train_data, train_results_matern, kernel_f=matern12_kernel, method_name="Matern12")

    print("Cauchy kernel")
    train_results_cauchy = fit_kpls(train_data, validation_data, kernel_f=cauchy_kernel, method_name="Cauchy")
    test_results_cauchy = eval_test(test_data, train_data, train_results_cauchy, kernel_f=cauchy_kernel, method_name="Cauchy")

    train_results_all = pd.concat(
        [train_results_rbf, train_results_laplac, train_results_cauchy],
        ignore_index=True
    )
    train_results_all.to_csv(dataset_name + "_train.csv")

    test_results_all = pd.concat(
        [test_results_rbf, test_results_laplac, test_results_cauchy],
        ignore_index=True
    )
    train_results_all.to_csv(dataset_name + "_test.csv")

    best_row_train = test_results_all.loc[test_results_all["rmse_test"].idxmin()]

    units_to_plot = [3,12,20]
    for unit_id in units_to_plot:
        plot_unit(train_data, test_data, best_row_train, dataset_name, unit_id)


train_data1, validation_data1, dropped_cols1 = train_from_file("./NASA-Turbofan-data/data/train_FD001.txt")
test_data1 = test_from_file("./NASA-Turbofan-data/data/test_FD001.txt", "./NASA-Turbofan-data/data/RUL_FD001.txt", dropped_cols=dropped_cols1)
train_data1, validation_data1, test_data1, mu_train1, sd_train1 = normalize_data(train_data1, validation_data1, test_data1)
process_dataset(train_data1, validation_data1, test_data1, "FD001")

train_data2, validation_data2, dropped_cols2 = train_from_file("./NASA-Turbofan-data/data/train_FD002.txt")
test_data2 = test_from_file("./NASA-Turbofan-data/data/test_FD002.txt", "./NASA-Turbofan-data/data/RUL_FD002.txt", dropped_cols=dropped_cols2)
train_data2, validation_data2, test_data2, mu_train2, sd_train2 = normalize_data(train_data2, validation_data2, test_data2)
process_dataset(train_data2, validation_data2, test_data2, "FD002")

train_data3, validation_data3, dropped_cols3 = train_from_file("./NASA-Turbofan-data/data/train_FD003.txt")
test_data3 = test_from_file("./NASA-Turbofan-data/data/test_FD003.txt", "./NASA-Turbofan-data/data/RUL_FD003.txt", dropped_cols=dropped_cols3)
train_data3, validation_data3, test_data3, mu_train3, sd_train3 = normalize_data(train_data3, validation_data3, test_data3)
process_dataset(train_data3, validation_data3, test_data3, "FD003")

train_data4, validation_data4, dropped_cols4 = train_from_file("./NASA-Turbofan-data/data/train_FD004.txt")
test_data4 = test_from_file("./NASA-Turbofan-data/data/test_FD004.txt", "./NASA-Turbofan-data/data/RUL_FD004.txt", dropped_cols=dropped_cols4)
train_data4, validation_data4, test_data4, mu_train4, sd_train4 = normalize_data(train_data4, validation_data4, test_data4)
process_dataset(train_data4, validation_data4, test_data4, "FD004")
