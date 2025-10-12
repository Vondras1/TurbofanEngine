# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display
import os

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


# %% PLS starts here -----------------------------------------------------------------------------------------------
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# %% Cross-Validation ----------------------------------------------------------------------------------------------
def pls_groupcv_press_q2(X, Y, groups, maxLV=10, n_splits=5):
    """
    Cross-Validation with respect to unit numbers (GroupKFold)

    groups - vector that have same length as y, contains unit ID. Used for GroupKFold
    maxLV - max number of latent variables
    n_splits - number of folds
    """
    gkf = GroupKFold(n_splits=n_splits)
    PRESS_folds = []
    Q2_folds = []


    for a in range(1, maxLV + 1):
        press_list, q2_list = [], []
        for train_idx, valid_idx in gkf.split(X, Y, groups=groups):
            Xtrain, Ytrain = X[train_idx], Y[train_idx]
            Xvalid, Yvalid = X[valid_idx], Y[valid_idx]

            scaler = StandardScaler()
            XtrainZ = scaler.fit_transform(Xtrain)
            XvalidZ = scaler.transform(Xvalid)

            r = min(a, XtrainZ.shape[1], max(1, min(XtrainZ.shape[0]-1, np.linalg.matrix_rank(XtrainZ))))
            pls = PLSRegression(n_components=r, scale=False)
            pls.fit(XtrainZ, Ytrain)

            Ypred = pls.predict(XvalidZ).ravel()
            press = np.sum((Yvalid - Ypred)**2)

            Ymean = float(np.mean(Ytrain))
            tss = np.sum((Yvalid - Ymean)**2)
            q2 = 1.0 - press / max(tss, np.finfo(float).eps)

            press_list.append(press)
            q2_list.append(q2)

        PRESS_folds.append(press_list)
        Q2_folds.append(q2_list)

    PRESS_folds = np.array(PRESS_folds).T   # shape: (folds, LV)
    Q2_folds    = np.array(Q2_folds).T

    PRESS_mean = np.nanmean(PRESS_folds, axis=0)
    Q2_mean    = np.nanmean(Q2_folds,    axis=0)

    press_best = int(np.nanargmin(PRESS_mean) + 1)
    q2_best    = int(np.nanargmax(Q2_mean)    + 1)

    return {
        "PRESS_folds": PRESS_folds,
        "Q2_folds": Q2_folds,
        "PRESS_mean": PRESS_mean,
        "Q2_mean": Q2_mean,
        "press_best": press_best,
        "q2_best": q2_best,
        "Keff": PRESS_folds.shape[0],
        "maxLV": PRESS_mean.size
    }

def plot_lv_curves(res):
    xs = np.arange(1, res["maxLV"] + 1)

    plt.figure(figsize=(7,4.2))
    plt.plot(xs, res["PRESS_mean"], "-o", linewidth=1.3); plt.grid(True)
    plt.xlabel("Number of latent variables"); plt.ylabel("mean PRESS$_{CV}$")
    plt.title(f"PRESS$_{{CV}}$ (mean, K={res['Keff']})")
    plt.axvline(res["press_best"], linestyle="--"); plt.show()

    plt.figure(figsize=(7,4.2))
    plt.plot(xs, res["Q2_mean"], "-o", linewidth=1.3); plt.grid(True)
    plt.xlabel("Number of latent variables"); plt.ylabel("mean Q$^2_{CV}$")
    plt.title(f"Q$^2_{{CV}}$ (mean, K={res['Keff']})")
    ymin = min(-0.2, float(np.min(res["Q2_mean"])) - 0.05)
    plt.ylim([ymin, 1.0]); plt.axvline(res["q2_best"], linestyle="--"); plt.show()


# X1 = train_data1.iloc[:, 2:-1]
# Y1 = train_data1.iloc[:, -1]
# g_train = train_data1.iloc[:, 0]

# res = pls_groupcv_press_q2(X1.to_numpy(), Y1.to_numpy(), g_train.to_numpy(), maxLV=10, n_splits=5)
# plot_lv_curves(res)
# print("Best by PRESS:", res["press_best"], "   Best by Q²:", res["q2_best"])


# %% Final training ---------------------------------------------------------------------------------
def predict_on_df(model, df, test=False):
    Xv = df.iloc[:, 2:-1]
    Yv = df.iloc[:, -1]
    Ypred = model.predict(Xv).ravel()
    return Yv, Ypred


def final_training(train_data, validation_data, test_data, prefix="", n_components=4, test_unit_id=20):
    """Train final PLS model, evaluate on validation and test sets and produce plots/files.

    Kept as close as possible to the original script but wrapped into a function.
    """
    # Normalize this dataset using training stats
    train_data, validation_data, test_data, mu_train, sd_train = normalize_data(train_data, validation_data, test_data)

    # train final model
    X1 = train_data.iloc[:, 2:-1]
    Y1 = train_data.iloc[:, -1]

    pls2 = PLSRegression(n_components=n_components, scale=False)
    pls2.fit(X1, Y1)

    # Validation predictions
    Yv, Ypred = predict_on_df(model=pls2, df=validation_data)
    Ypred = np.clip(Ypred, 0, None)

    # Yv and Ypred side by side
    comparison = pd.DataFrame({
        "Y_true": Yv.values,
        "Y_pred": Ypred
    })
    display(comparison)
    results_dir = "Results"
    comp_name = f"{prefix}_comparison_results.csv" if prefix else "comparison_results.csv"
    comparison.to_csv(os.path.join(results_dir, comp_name), index=False)

    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(Yv, Ypred))
    mae = mean_absolute_error(Yv, Ypred)
    r2 = r2_score(Yv, Ypred)

    print(f"Validation dataset metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

    # window = 25: RMSE=33.651, MAE=26.502, R²=0.737
    # window = 20: RMSE=34.457, MAE=27.083, R²=0.725
    # window = 15: RMSE=35.282, MAE=27.628, R²=0.711
    # window = 10: RMSE=35.997, MAE=28.062, R²=0.699

    # Validation unit plot
    print(validation_data["unit number"].unique())
    unit_id = validation_data["unit number"].unique()[0]
    val_unit = validation_data[validation_data["unit number"] == unit_id]

    # Prepare inputs
    print(f"Val unit {unit_id} shape: {val_unit.shape}")
    Xunit = val_unit.iloc[:, 2:-1].to_numpy()
    Yunit = val_unit["RUL"].to_numpy()

    # Predict
    print(f"Xunit shape: {Xunit.shape}")
    Ypred = pls2.predict(Xunit).ravel()
    Ypred = np.clip(Ypred, 0, None)

    # Compute residuals
    residuals = Yunit - Ypred

    # Metrics for unit
    rmse = np.sqrt(mean_squared_error(Yunit, Ypred))
    mae = mean_absolute_error(Yunit, Ypred)
    r2 = r2_score(Yunit, Ypred)

    print(f"Unit {unit_id} metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

    # Plot residuals for that unit
    plt.figure(figsize=(14, 4))
    plt.plot(residuals, '-', linewidth=1.2)
    plt.xlabel("Cycle index (time)")
    plt.ylabel("Residual")
    plt.title(f"Validation - Residuals for Unit {unit_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

    # Predict on test
    Yt, Ypred = predict_on_df(model=pls2, df=test_data)
    Ypred = np.clip(Ypred, 0, None)

    # Y and Ypred side by side for test
    test_comparison = pd.DataFrame({
        "Y_true": Yt.values,
        "Y_pred": Ypred
    })
    display(test_comparison)
    test_name = f"{prefix}_test_comparison_results.csv" if prefix else "test_comparison_results.csv"
    test_comparison.to_csv(os.path.join(results_dir, test_name), index=False)

    rmse = np.sqrt(mean_squared_error(Yt, Ypred))
    mae = mean_absolute_error(Yt, Ypred)
    r2 = r2_score(Yt, Ypred)

    print(f"Test dataset metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

    # Test unit plot
    unit_id = test_unit_id
    test_unit = test_data[test_data["unit number"] == unit_id]

    # Prepare inputs
    Xunit = test_unit.iloc[:, 2:-1].to_numpy()
    Yunit = test_unit["RUL"].to_numpy()

    # Predict
    Ypred = pls2.predict(Xunit).ravel()
    Ypred = np.clip(Ypred, 0, None)

    # Compute residuals
    residuals = Yunit - Ypred

    # Metrics for test unit
    rmse = np.sqrt(mean_squared_error(Yunit, Ypred))
    mae = mean_absolute_error(Yunit, Ypred)
    r2 = r2_score(Yunit, Ypred)

    print(f"Unit {unit_id} metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

    # Plot residuals for that unit and save
    plt.figure(figsize=(14, 4))
    plt.plot(residuals, '-', linewidth=1.2)
    plt.xlabel("Cycle index (time)")
    plt.ylabel("Residual")
    plt.title(f"TEST - Residuals for Unit {unit_id}")
    plt.grid(True)
    plt.tight_layout()
    res_fname = f"residuals_test_unit_{unit_id}_{prefix}.png" if prefix else f"residuals_test_unit_{unit_id}.png"
    plt.savefig(os.path.join(results_dir, res_fname), dpi=150)

    # Observed vs Predicted (scatter) and save
    plt.figure(figsize=(5, 5))
    plt.scatter(Yunit, Ypred, s=15, c="tab:blue", alpha=0.7)
    plt.plot([Yunit.min(), Yunit.max()], [Yunit.min(), Yunit.max()], "k--", lw=1.2)
    plt.xlabel("Observed RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"TEST - Observed vs Predicted (Unit {unit_id})")
    plt.grid(True)
    plt.axis("equal")
    obs_fname = f"obs_vs_pred_test_unit_{unit_id}_{prefix}.png" if prefix else f"obs_vs_pred_test_unit_{unit_id}.png"
    plt.savefig(os.path.join(results_dir, obs_fname), dpi=150)


def main():
    # Run final analysis for each dataset

    train_data1, validation_data1, dropped_cols1 = train_from_file("./NASA-Turbofan-data/data/train_FD001.txt")
    test_data1 = test_from_file("./NASA-Turbofan-data/data/test_FD001.txt", "./NASA-Turbofan-data/data/RUL_FD001.txt", dropped_cols=dropped_cols1)

    train_data2, validation_data2, dropped_cols2 = train_from_file("./NASA-Turbofan-data/data/train_FD002.txt")
    test_data2 = test_from_file("./NASA-Turbofan-data/data/test_FD002.txt", "./NASA-Turbofan-data/data/RUL_FD002.txt", dropped_cols=dropped_cols2)

    train_data3, validation_data3, dropped_cols3 = train_from_file("./NASA-Turbofan-data/data/train_FD003.txt")
    test_data3 = test_from_file("./NASA-Turbofan-data/data/test_FD003.txt", "./NASA-Turbofan-data/data/RUL_FD003.txt", dropped_cols=dropped_cols3)

    train_data4, validation_data4, dropped_cols4 = train_from_file("./NASA-Turbofan-data/data/train_FD004.txt")
    test_data4 = test_from_file("./NASA-Turbofan-data/data/test_FD004.txt", "./NASA-Turbofan-data/data/RUL_FD004.txt", dropped_cols=dropped_cols4)
    
    datasets = [
        ("FD001", train_data1, validation_data1, test_data1),
        ("FD002", train_data2, validation_data2, test_data2),
        ("FD003", train_data3, validation_data3, test_data3),
        ("FD004", train_data4, validation_data4, test_data4),
    ]

    for prefix, train_d, val_d, test_d in datasets:
        print(f"\n--- Running cross-validation (pls_groupcv_press_q2) for {prefix} ---")
        # normalize a copy for CV so we don't mutate the original datasets
        t_copy = train_d.copy()
        v_copy = val_d.copy()
        te_copy = test_d.copy()
        t_copy, v_copy, te_copy, mu_t, sd_t = normalize_data(t_copy, v_copy, te_copy)

        X_cv = t_copy.iloc[:, 2:-1]
        Y_cv = t_copy.iloc[:, -1]
        g_cv = t_copy.iloc[:, 0]

        res = pls_groupcv_press_q2(X_cv.to_numpy(), Y_cv.to_numpy(), g_cv.to_numpy(), maxLV=10, n_splits=5)
        plot_lv_curves(res)
        print(f"Best by PRESS: {res['press_best']}   Best by Q²: {res['q2_best']}")

        print(f"\n--- Running final training for {prefix} ---")
        final_training(train_d, val_d, test_d, prefix=prefix)


if __name__ == "__main__":
    main()


# %%
# %%
