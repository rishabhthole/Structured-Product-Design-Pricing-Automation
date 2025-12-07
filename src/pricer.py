# src/pricer.py
import numpy as np
from scipy.optimize import brentq
from .simulator import correlated_default_sim
from .utils import pv

def payoff_given_defaults(default_matrix, face=1.0, coupon=0.08, tenor=2, recovery=0.4):
    """
    Define payoff rule:
    - If no defaults -> principal + coupon*tenor
    - If 1 default -> principal reduced by 1/len * (1 - recovery) of that portion
    - If 2+ defaults -> principal reduced accordingly (or zero)
    For simplicity we compute final cash as: remaining principal + coupon*tenor (coupon paid if surviving)
    """
    n_names = default_matrix.shape[1]
    # number of defaults per simulation
    num_defaults = default_matrix.sum(axis=1)
    # loss per default (proportional) on that name: (1 - recovery) * (face / n_names)
    loss_per_default = (1 - recovery) * (face / n_names)
    principal_remaining = np.maximum(0.0, face - num_defaults * loss_per_default)
    # assume coupons are paid fully only if corresponding notional survived (simplified)
    coupon_paid = coupon * tenor * (principal_remaining / face)
    final_val = principal_remaining + coupon_paid
    return final_val

def montecarlo_price(pd_annual, correlation, tenor, coupon, num_sims, discount_rate, recovery, face=1.0, rng=None):
    defaults, cum_pd = correlated_default_sim(pd_annual, correlation, tenor, num_sims, rng=rng)
    final_vals = payoff_given_defaults(defaults, face=face, coupon=coupon, tenor=tenor, recovery=recovery)
    # discount final value back to present at discount_rate (annual)
    pv_vals = final_vals / ((1 + discount_rate) ** tenor)
    return pv_vals.mean(), pv_vals.std()

def fair_coupon_for_target_yield(target_yield, pd_annual, correlation, tenor, num_sims, discount_rate, recovery, face=1.0, rng=None):
    """Find coupon such that price equals fair present value implied by target yield."""
    def objective(c):
        price, _ = montecarlo_price(pd_annual, correlation, tenor, c, num_sims, discount_rate, recovery, face, rng=rng)
        # target price given target yield = face / (1 + target_yield)**tenor
        target_price = face / ((1 + target_yield) ** tenor)
        return price - target_price

    # search coupon between -0.5 and 1.0 (reasonable)
    try:
        fair_c = brentq(objective, -0.5, 1.0, maxiter=50)
    except ValueError:
        # if root not bracketed, fallback to optimizing for minimal difference
        c_grid = np.linspace(0, 0.5, 51)
        diffs = [abs(montecarlo_price(pd_annual, correlation, tenor, c, num_sims, discount_rate, recovery, face, rng=rng)[0] - face / ((1 + target_yield) ** tenor)) for c in c_grid]
        fair_c = c_grid[int(np.argmin(diffs))]
    return fair_c
