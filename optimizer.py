
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.preprocessing import StandardScaler, KernelCenterer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn import model_selection
from scipy.optimize import minimize_scalar
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
import numpy as np
from kernel import cauchy_kernel, cauchy_gamma_bounds, matern12_kernel

def gamma_bounds_from_data(X, kernel_f=rbf_kernel, spread=3, tiny=1e-300):
    if kernel_f == rbf_kernel:
        # Euclidean distances for RBF (note: rbf uses squared Euclidean in the exponent)
        d = pairwise_distances(X, metric='euclidean')
        nz = d[d > 0]
        if nz.size == 0:
            return 1e-6, 1.0
        m = np.median(nz)
        gamma0 = 1.0 / (2.0 * (m**2) + 1e-12)

    elif kernel_f == laplacian_kernel:
        # L1 distances for Laplacian
        d1 = pairwise_distances(X, metric='manhattan')
        nz = d1[d1 > 0]
        if nz.size == 0:
            return 1e-6, 1.0
        m = np.median(nz)
        gamma0 = 1.0 / (m + 1e-12)
    elif kernel_f == cauchy_kernel:
        return cauchy_gamma_bounds(X, spread)
    elif kernel_f == matern12_kernel:
        return 1e-6, 1e+2
    else:
        raise ValueError("Unsupported kernel")

    g_low  = gamma0 * (10 ** -spread)
    g_high = gamma0 * (10 **  spread)
    return g_low, g_high

def gamma_bounds_from_blocks(blocks):
    # collect medians of nonzero distances per fold
    meds = []
    for b in blocks:
        d = b["D_tr"]
        nz = d[d > 0]
        if nz.size:
            meds.append(np.median(nz))
    if not meds:
        return 1e-6, 1.0  # fallback
    m = np.median(meds)
    # gamma ≈ 1 / (2*m); give a decade or two around it
    g0 = 1.0 / (2.0 * m + 1e-12)
    g_low  = g0 * 1e-3
    g_high = g0 * 1e+3
    return g_low, g_high

def center_kernel(K_tr, K_val):
    """
    Center K_tr and K_val using training statistics.
    K_tr: (n x n), K_val: (m x n)
    Returns: K_tr_c, K_val_c
    """
    # training means
    mean_rows_tr = K_tr.mean(axis=1, keepdims=True)      # (n,1)
    mean_cols_tr = K_tr.mean(axis=0, keepdims=True)      # (1,n)
    grand_tr     = K_tr.mean()                            # scalar

    # center train
    K_tr_c = K_tr - mean_rows_tr - mean_cols_tr + grand_tr

    # center val (note: use training column means; row means from val)
    mean_rows_val = K_val.mean(axis=1, keepdims=True)    # (m,1)
    # reuse mean_cols_tr and grand_tr from training
    K_val_c = K_val - mean_rows_val - mean_cols_tr + grand_tr
    return K_tr_c, K_val_c

def optimize(data, max_lv=20, folds=8, show_plot=True, kernel_f=rbf_kernel, method_name = "RBF_kernel"):
    groups = data["unit number"].values
    X = data.iloc[:, 2:-1]
    y = data.iloc[:, -1]
    
    gkf = GroupKFold(n_splits=folds)
    splits = list(gkf.split(X, y, groups=groups))
    gamma_low, gamma_high = gamma_bounds_from_data(X, kernel_f)
    print(f"Gamma bounds: ({gamma_low}; {gamma_high})")

    results = []
    def neg_q2(gamma, n_lv):
        q2_scores = []

        for train_idx, test_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx] 
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx] 
            K_tr = kernel_f(X_tr, X_tr, gamma=gamma) 

            centerer = KernelCenterer() 
            K_tr_c = centerer.fit_transform(K_tr) 
            K_val = kernel_f(X_val, X_tr, gamma=gamma) 
            K_val_c = centerer.transform(K_val) 

            # PLS regression on kernel-transformed features 
            pls = PLSRegression(n_components=n_lv) 
            pls.fit(K_tr_c, y_tr) 
            y_pred = pls.predict(K_val_c) 

            # Compute Q^2 for this fold (1 - PRESS/TSS) 
            ss_res = np.sum((y_val - y_pred.ravel())**2) 
            ss_tot = np.sum((y_val - np.mean(y_tr))**2)
            q2_fold = 1 - ss_res/ss_tot 
            q2_scores.append(q2_fold)

        return -np.mean(q2_scores)
    
    def solve_one(n_lv):
        # bounded scalar minimization for this n_lv
        res = minimize_scalar(lambda g: neg_q2(g, n_lv),
                              bounds=(gamma_low, gamma_high),
                              method='bounded',
                              options={'xatol': 1e-3})
        print(f"n_lv: {n_lv}, gamma: {res.x}, q2: {-res.fun}")
        return {"n_lv": n_lv, "gamma": res.x, "q2": -res.fun}
    
    print("Optimizer starts...")
    results = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(solve_one)(n_lv) for n_lv in range(1, max_lv + 1)
    )

    results = sorted(results, key=lambda d: d["n_lv"])

    # to a DataFrame for convenience
    df_results = pd.DataFrame(results)

    if show_plot:
        plt.figure()
        plt.plot(df_results["n_lv"], df_results["q2"], marker="o")
        plt.xlabel("Number of latent variables (n_lv)")
        plt.ylabel("CV $Q^2$")
        plt.title(f"{method_name} CV $Q^2$ vs n_lv, K_folds={folds}")
        plt.grid(True)
        plt.show()

    return df_results
    