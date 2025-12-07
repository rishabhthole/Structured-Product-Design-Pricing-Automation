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

# def montecarlo_price(pd_annual, correlation, tenor, coupon, num_sims, discount_rate, recovery, face=1.0, rng=None):
#     defaults, cum_pd = correlated_default_sim(pd_annual, correlation, tenor, num_sims, rng=rng)
#     final_vals = payoff_given_defaults(defaults, face=face, coupon=coupon, tenor=tenor, recovery=recovery)
#     # discount final value back to present at discount_rate (annual)
#     pv_vals = final_vals / ((1 + discount_rate) ** tenor)
#     return pv_vals.mean(), pv_vals.std()

def montecarlo_price(pd_annual, correlation, tenor, coupon, num_sims,
                     discount_curve_maturities=None, discount_curve_rates=None,
                     recovery=0.4, periods_per_year=4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    # Call simulator correctly
    defaults, cum_pd = correlated_default_sim(pd_annual, correlation, tenor, num_sims, rng=rng)
    
    # Compute PVs
    losses = defaults.sum(axis=1) / len(pd_annual)
    expected_recovery = (1 - losses) * (1 - recovery)
    discounted_cf = (1 + coupon * tenor) * expected_recovery
    pv = np.mean(discounted_cf)
    std = np.std(discounted_cf)
    
    return pv, std


def fair_coupon_for_par(pd_annual, correlation, tenor, num_sims,
                        discount_rate=0.03, recovery=0.4, rng=None):
    """
    Solve for the coupon that prices the note at par (PV = 1.0)
    using Monte Carlo simulation and Brent’s root finding.
    """
    from src.pricer import montecarlo_price  # local import to avoid circular dependency

    if rng is None:
        rng = np.random.default_rng()

    def price_minus_par(coupon):
        pv, _ = montecarlo_price(
            pd_annual=pd_annual,
            correlation=correlation,
            tenor=tenor,
            coupon=coupon,
            num_sims=num_sims,
            discount_rate=discount_rate,
            recovery=recovery,
            rng=None
        )
        return pv - 1.0

    try:
        # Solve for coupon between -50% and +50%
        fair_coupon = brentq(price_minus_par, -0.5, 0.5)
    except ValueError:
        # Fallback linear approximation if no root found
        pv0, _ = montecarlo_price(
            pd_annual, correlation, tenor, 0.0,
            num_sims, discount_rate=discount_rate, recovery=recovery, rng=None
        )
        fair_coupon = (1 / pv0 - 1) / tenor

    return fair_coupon
