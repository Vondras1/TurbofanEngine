
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process.kernels import Matern

## Cauchy
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

def matern12_kernel(X, Y=None, gamma=1.0):
    ls = 1.0 / (gamma + 1e-12)
    k = Matern(length_scale=ls, nu=0.5)
    return k(X, X if Y is None else Y)