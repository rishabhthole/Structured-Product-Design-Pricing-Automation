# src/simulator.py
import numpy as np
from scipy.stats import norm
from .utils import build_cov_matrix

def correlated_default_sim(pd_annual, correlation, tenor, num_sims, rng=None):
    """
    Simulate default outcomes across tenor years (simplified: one check at maturity).
    Returns: defaults matrix shape (num_sims, n_names) with 1 if default occurred over tenor.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(pd_annual)
    cov = build_cov_matrix(n, correlation)
    L = np.linalg.cholesky(cov)

    # Convert annual PD to cumulative PD over tenor assuming constant hazard:
    # cumulative_PD = 1 - exp(-lambda * tenor), lambda = -ln(1-annual_pd)
    lambdas = -np.log(1 - np.array(pd_annual))
    cum_pd = 1 - np.exp(-lambdas * tenor)

    # Map cum_pd to Gaussian threshold
    thresholds = norm.ppf(cum_pd)  # quantiles such that P(Z < thresh) = cum_pd

    Z = rng.standard_normal(size=(num_sims, n))
    correlated = Z.dot(L.T)
    defaults = (correlated < thresholds).astype(int)
    return defaults, cum_pd

# Optional: simulate time-of-default (first passage) — omitted for simplicity
