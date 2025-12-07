"""
CLN Simulator Final
Single-file end-to-end simulator for a multi-name Credit-Linked Note (CLN).
Features:
- periodic coupon flows (quarterly by default)
- correlated defaults via Gaussian copula (per-period checks)
- fair-coupon calibration (price = par)
- yield curve discounting
- benchmark comparison and sensitivity runs
- Excel export and charts

Dependencies:
numpy, pandas, scipy, matplotlib, openpyxl
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os
from openpyxl import Workbook

# -------------------------
# Utilities
# -------------------------
def build_cov_matrix(n, rho):
    mat = np.full((n, n), rho)
    np.fill_diagonal(mat, 1.0)
    return mat

def interp_discount_factors(maturities, zero_rates, payment_times):
    """
    Simple zero curve interpolation (annual compounding assumption).
    maturities: list of tenor points (years) for zero_rates
    zero_rates: zero rates corresponding to maturities (annual)
    payment_times: array of times to discount
    Returns discount factors for each payment_time
    """
    # linear interpolation of zero rates
    zr = np.interp(payment_times, maturities, zero_rates)
    dfs = 1.0 / ((1 + zr) ** payment_times)
    return dfs

def annual_pd_to_period_cum_pd(annual_pd, period_length):
    """
    Convert annual PD to cumulative PD over period_length years (assuming constant hazard).
    annual_pd: array-like of annual pd (e.g., 0.02)
    period_length: length of period in years (e.g., 0.25)
    Returns cumulative PD per period.
    """
    lam = -np.log(1 - np.array(annual_pd))
    cum_pd = 1 - np.exp(-lam * period_length)
    return cum_pd

# -------------------------
# Core simulation (per-period default checks)
# -------------------------
def simulate_defaults_timeline(pd_annual, correlation, tenor, periods_per_year, num_sims, rng=None):
    """
    Simulate correlated defaults across multiple discrete periods (e.g., quarterly checks).
    Method: For each period, generate correlated standard normals; if variable < threshold, default occurs in that period.
    Returns:
      default_matrix: shape (num_sims, n_names, n_periods) with 1 if default in that period (first default only)
      default_time: array (num_sims, n_names) = period index of default (n_periods means no default)
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(pd_annual)
    total_periods = int(tenor * periods_per_year)
    period_length = 1.0 / periods_per_year

    # cumulative PD for each single period
    period_pd = annual_pd_to_period_cum_pd(pd_annual, period_length)  # per-period marginal default prob
    # Convert marginal PD to Gaussian threshold for each period (same threshold every period under constant hazard)
    thresholds = norm.ppf(period_pd)  # threshold such that P(Z<threshold) = period_pd

    cov = build_cov_matrix(n, correlation)
    L = np.linalg.cholesky(cov)

    # Prepare outputs
    defaults_matrix = np.zeros((num_sims, n, total_periods), dtype=int)
    default_time = np.full((num_sims, n), fill_value=total_periods, dtype=int)  # default_time==total_periods -> survived

    # For each period, simulate correlated normals and assign defaults only if not already defaulted
    for t in range(total_periods):
        Z = rng.standard_normal(size=(num_sims, n))
        correlated = Z.dot(L.T)
        # default this period if correlated < thresholds
        period_defaults = (correlated < thresholds).astype(int)
        # But only mark defaults for names that haven't defaulted earlier
        for i in range(n):
            not_yet = (default_time[:, i] == total_periods)
            new_defaults_idx = np.where((period_defaults[:, i] == 1) & not_yet)[0]
            if new_defaults_idx.size > 0:
                defaults_matrix[new_defaults_idx, i, t] = 1
                default_time[new_defaults_idx, i] = t  # record first-default period index

    return defaults_matrix, default_time, total_periods

# -------------------------
# Cashflow & payoff calculation
# -------------------------
def compute_payoff_from_defaults(default_time, face=1.0, coupon=0.08, tenor=2, periods_per_year=4, recovery=0.4, discount_times=None, dfs=None):
    """
    Given default times per name per simulation, compute PV of cashflows at t=0.
    - coupon is annual rate. Coupons paid per period on surviving notional fraction.
    - on default in period t, assume default occurs at period end and recovery applied to that period's notional (simplification).
    Parameters:
      default_time: shape (num_sims, n_names) period index of first default (total_periods means survived)
      discount_times: list/array of payment times (years) corresponding to coupon and final principal
      dfs: discount factors for each payment time (if None we will not discount)
    Returns:
      pv_vals: array (num_sims,) present value of the note for each simulation
      raw_cashflows: optional (not returned) - could be extended
    """
    num_sims, n = default_time.shape
    total_periods = int(tenor * periods_per_year)
    period_length = 1.0 / periods_per_year
    coupon_per_period = coupon * period_length

    pv_vals = np.zeros(num_sims)

    # Payment times: coupons at 1..total_periods, final principal at period total_periods
    payment_times = np.array([(t+1) * period_length for t in range(total_periods)])  # years
    if discount_times is None:
        discount_times = payment_times.copy()
    if dfs is None:
        # no discounting (not recommended), set dfs=1 for all
        dfs = np.ones_like(discount_times)

    for sim in range(num_sims):
        # surviving fraction of notional per name at each period
        # for simplicity: equal notionals per name = face / n
        notional_per_name = face / n
        # for each period, calculate coupon paid = sum of surviving notionals * coupon_per_period
        cf_pvs = 0.0
        for p_idx, pay_t in enumerate(payment_times):
            # determine surviving notional at period p_idx (i.e., those with default_time > p_idx)
            surviving_mask = (default_time[sim, :] > p_idx)  # if default_time == p_idx => default happens this period -> assume default before coupon at period end? We assume coupon accrued up to default and paid proportionally; to keep simple: pay coupon only if surviving at period end
            surviving_notional = surviving_mask.sum() * notional_per_name
            coupon_cf = surviving_notional * coupon_per_period
            # At final period also principal repayment for surviving notional
            principal_cf = 0.0
            if p_idx == total_periods - 1:
                principal_cf = surviving_notional
            # Add recoveries for names defaulted in this period: for defaulted names, recovery applied at same period as default
            defaulted_mask = (default_time[sim, :] == p_idx)
            defaulted_count = defaulted_mask.sum()
            recovery_cf = defaulted_count * notional_per_name * recovery
            # total CF at this period
            total_cf = coupon_cf + principal_cf + recovery_cf
            # discount
            df = dfs[p_idx]
            cf_pvs += total_cf * df
        pv_vals[sim] = cf_pvs
    return pv_vals

# -------------------------
# Monte Carlo wrapper and fair coupon solver
# -------------------------
def montecarlo_price(pd_annual, correlation, tenor, coupon, num_sims, discount_curve_maturities, discount_curve_rates, recovery=0.4, periods_per_year=4, rng=None):
    """
    Run full simulation and return PV distribution and summary stats.
    """
    defaults_matrix, default_time, total_periods = simulate_defaults_timeline(pd_annual, correlation, tenor, periods_per_year, num_sims, rng=rng)
    payment_times = np.array([(t+1) * (1.0/periods_per_year) for t in range(total_periods)])
    dfs = interp_discount_factors(discount_curve_maturities, discount_curve_rates, payment_times)
    pv_vals = compute_payoff_from_defaults(default_time, face=1.0, coupon=coupon, tenor=tenor, periods_per_year=periods_per_year, recovery=recovery, discount_times=payment_times, dfs=dfs)
    return pv_vals, default_time

def fair_coupon_for_par(pd_annual, correlation, tenor, num_sims, discount_curve_maturities, discount_curve_rates, recovery=0.4, periods_per_year=4, tol=1e-4, rng=None):
    """
    Solve for coupon such that the simulated PV equals par (1.0)
    """
    def objective(c):
        pv_vals, _ = montecarlo_price(pd_annual, correlation, tenor, c, num_sims, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year, rng=rng)
        price = pv_vals.mean()
        return price - 1.0  # want price == par

    # bracket reasonable coupon range
    low, high = -0.5, 1.0
    try:
        fair_c = brentq(objective, low, high, xtol=tol, maxiter=60)
    except Exception:
        # fallback to grid search if root finding fails
        grid = np.linspace(0.0, 0.4, 41)
        prices = []
        for c in grid:
            pv_vals, _ = montecarlo_price(pd_annual, correlation, tenor, c, num_sims//5, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year, rng=rng)  # fewer sims for speed
            prices.append(pv_vals.mean())
        idx = np.argmin(np.abs(np.array(prices) - 1.0))
        fair_c = grid[idx]
    return fair_c

# -------------------------
# Benchmark comparison
# -------------------------
def equivalent_bond_yield(face=1.0, coupon=0.0, tenor=2, discount_curve_maturities=None, discount_curve_rates=None):
    """
    Compute yield of a plain bond that would have same PV as given discount curve (for benchmark purpose).
    We'll return the YTM required to price the bond at par given the discount curve as benchmark risk-free.
    """
    # For simplicity: compute fair annual yield such that PV(coupon payments + principal)=1 given flat yield y.
    # We can approximate by solving for y using discount_curve as reference; but simplest: compute spot PV of unit principal discounted by given curve then derive spread required.
    payment_times = np.array([i+1 for i in range(int(tenor))])  # annual coupons assumed for benchmark
    ref_dfs = interp_discount_factors(discount_curve_maturities, discount_curve_rates, payment_times)
    # Suppose bond pays annual coupon r; price = r * sum(df) + 1 * df[-1] ; set price = 1 solve for r
    sum_dfs = ref_dfs.sum()
    r = (1.0 - ref_dfs[-1]) / sum_dfs
    return r

# -------------------------
# Reporting: charts & excel
# -------------------------
def plot_payoff_distribution(pv_vals, outdir, fname_prefix="payoff"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(pv_vals, bins=80)
    plt.title("Distribution of Note PV (per simulation)")
    plt.xlabel("PV at t=0")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = os.path.join(outdir, f"{fname_prefix}_hist.png")
    plt.savefig(hist_path)
    plt.close()

    # CDF
    plt.figure(figsize=(8,5))
    sorted_vals = np.sort(pv_vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    plt.plot(sorted_vals, cdf)
    plt.title("CDF of Note PV")
    plt.xlabel("PV at t=0")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    cdf_path = os.path.join(outdir, f"{fname_prefix}_cdf.png")
    plt.savefig(cdf_path)
    plt.close()
    return hist_path, cdf_path

def save_summary_excel(outpath, params, price_mean, price_std, fair_coupon, extra_table=None):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Parameter", "Value"])
    for k,v in params.items():
        ws.append([k, str(v)])
    ws.append([])
    ws.append(["Monte Carlo Mean PV", price_mean])
    ws.append(["Monte Carlo PV StdDev", price_std])
    ws.append(["Fair Coupon (annual)", fair_coupon])
    if extra_table is not None:
        ws2 = wb.create_sheet("Extra")
        ws2.append(list(extra_table.columns))
        for row in extra_table.itertuples(index=False):
            ws2.append(list(row))
    wb.save(outpath)

# -------------------------
# Sensitivity wrapper
# -------------------------
def run_sensitivity(pd_annual, base_corr, tenor, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year, num_sims, corr_grid=None, pd_shock=None, outdir="outputs"):
    if corr_grid is None:
        corr_grid = np.linspace(0.0, 0.75, 6)
    if pd_shock is None:
        pd_shock = [0.9, 1.0, 1.1]  # multiplicative shocks
    results = []
    for corr in corr_grid:
        for shock in pd_shock:
            pd_shocked = list(np.array(pd_annual) * shock)
            # quick approximate fair coupon (fewer sims)
            fair_c = fair_coupon_for_par(pd_shocked, corr, tenor, max(10000, num_sims//10), discount_curve_maturities, discount_curve_rates, recovery, periods_per_year)
            pv_vals, _ = montecarlo_price(pd_shocked, corr, tenor, fair_c, max(10000, num_sims//10), discount_curve_maturities, discount_curve_rates, recovery, periods_per_year)
            results.append({
                "corr": corr,
                "pd_shock": shock,
                "fair_coupon": fair_c,
                "pv_mean": pv_vals.mean(),
                "pv_std": pv_vals.std()
            })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "sensitivity_table.csv"), index=False)
    return df

# -------------------------
# Example runner (main)
# -------------------------
if __name__ == "__main__":
    # Example parameters (you should change as needed)
    names = ["JPM", "DB", "HSBC"]
    pd_annual = [0.02, 0.03, 0.015]    # annual PDs
    correlation = 0.35
    tenor = 2
    periods_per_year = 4  # quarterly
    num_sims = 50000
    recovery = 0.4

    # Discount curve example (maturities in years and zero rates)
    discount_curve_maturities = [0.25, 0.5, 1, 2, 5, 10]
    discount_curve_rates = [0.005, 0.007, 0.01, 0.02, 0.03, 0.035]  # example zero rates

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # 1) Calibrate fair coupon (this is the main compute step)
    print("Calibrating fair coupon to par... (this may take time depending on num_sims)")
    fair_coupon = fair_coupon_for_par(pd_annual, correlation, tenor, num_sims, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year)
    print(f"Fair annual coupon (calibrated): {fair_coupon:.6f}")

    # 2) Run full Monte Carlo at calibrated coupon
    pv_vals, default_time = montecarlo_price(pd_annual, correlation, tenor, fair_coupon, num_sims, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year)
    mean_pv = pv_vals.mean()
    std_pv = pv_vals.std()
    print(f"Monte Carlo mean PV: {mean_pv:.6f}, std: {std_pv:.6f}")

    # 3) Plots
    hist_path, cdf_path = plot_payoff_distribution(pv_vals, outdir)

    # 4) Default count distribution
    default_counts = (default_time < tenor * periods_per_year).sum(axis=1)  # number of names defaulted per sim
    dc_series = pd.Series(default_counts)
    dc_table = dc_series.value_counts().sort_index().rename_axis("num_defaults").reset_index(name="frequency")
    dc_table["prob"] = dc_table["frequency"] / num_sims
    print("Default count probs (sample):")
    print(dc_table)

    # 5) Benchmark (simple)
    benchmark_yield = equivalent_bond_yield(tenor=tenor, discount_curve_maturities=discount_curve_maturities, discount_curve_rates=discount_curve_rates)
    print(f"Benchmark (approx) annual yield for par under curve: {benchmark_yield:.6f}")
    print(f"Structured note fair coupon: {fair_coupon:.6f} vs Benchmark yield: {benchmark_yield:.6f} => spread ~ {(fair_coupon - benchmark_yield):.6f}")

    # 6) Save Excel summary
    params = {
        "names": ",".join(names),
        "pd_annual": str(pd_annual),
        "correlation": correlation,
        "tenor": tenor,
        "periods_per_year": periods_per_year,
        "num_sims": num_sims,
        "recovery": recovery
    }
    excel_path = os.path.join(outdir, "cln_summary.xlsx")
    extra_table = pd.DataFrame({
        "metric": ["mean_pv", "std_pv", "fair_coupon", "benchmark_yield", "spread"],
        "value": [mean_pv, std_pv, fair_coupon, benchmark_yield, fair_coupon - benchmark_yield]
    })
    save_summary_excel(excel_path, params, mean_pv, std_pv, fair_coupon, extra_table=extra_table)
    print(f"Saved outputs to {outdir} (plots and Excel).")

    # 7) Run a quick sensitivity and save
    sens_df = run_sensitivity(pd_annual, correlation, tenor, discount_curve_maturities, discount_curve_rates, recovery, periods_per_year, num_sims, corr_grid=np.linspace(0.0,0.6,4), pd_shock=[0.9,1.0,1.1], outdir=outdir)
    print("Sensitivity run saved.")
