import numpy as np
from sklearn.metrics import pairwise_distances

def cauchy_gamma_bounds(X, spread=3):
    X = np.asarray(X)
    d = pairwise_distances(X, metric="euclidean")
    i, j = np.triu_indices_from(d, k=1)
    vals = d[i, j]
    nz = vals[vals > 0]
    if nz.size == 0:
        return 1e-6, 1.0
    m = np.median(nz)
    gamma0 = 1.0 / (m**2 + 1e-12)
    return gamma0 * 10**(-spread), gamma0 * 10**spread

def cauchy_kernel(X, Y=None, gamma=1.0):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    d2 = pairwise_distances(X, Y, metric="euclidean", squared=True)
    return 1.0 / (1.0 + gamma * d2)    

def polynomial_kernel(X, Y=None, gamma=None, coef0=1.0, degree=3):
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + coef0) ** degree