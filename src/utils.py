# src/utils.py
import numpy as np

def ann_to_period_pd(annual_pd, periods_per_year=1, tenor=1):
    """Convert annual PD to period PD for given tenor (assuming constant hazard)."""
    # convert to cumulative survival assumption: S(t) = exp(-lambda * t)
    lam = -np.log(1 - np.array(annual_pd))
    # period PD for 1 period of length tenor (if periods_per_year>1, adapt)
    period_pd = 1 - np.exp(-lam * (tenor))
    return period_pd

def pv(amount, t, r):
    """Present value of amount paid at time t using continuous-ish discounting (or flat r)."""
    return amount / ((1 + r) ** t)

def build_cov_matrix(n, rho):
    """Return an n x n covariance matrix with rho off-diagonal and 1 on diagonal."""
    mat = np.full((n, n), rho)
    np.fill_diagonal(mat, 1.0)
    return mat
