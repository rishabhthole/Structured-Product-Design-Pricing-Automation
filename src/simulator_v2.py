# src/simulator.py
import numpy as np
from scipy.stats import norm
from .utils import build_cov_matrix

def correlated_default_sim(pd_annual, correlation, tenor, num_sims, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(pd_annual)
    
    # Generate correlated standard normal samples
    corr_matrix = np.full((n, n), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    L = np.linalg.cholesky(corr_matrix)
    Z = rng.standard_normal(size=(num_sims, n))   # ✅ requires rng to be Generator
    correlated_Z = Z @ L.T

    # Convert to uniform via CDF
    U = 0.5 * (1 + np.erf(correlated_Z / np.sqrt(2)))

    # Determine defaults
    thresholds = 1 - np.exp(-np.array(pd_annual) * tenor)
    defaults = (U < thresholds).astype(int)
    
    return defaults, thresholds
