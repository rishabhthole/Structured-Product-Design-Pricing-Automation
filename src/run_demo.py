import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_PATH = os.path.join(BASE_DIR, "outputs", "example_report_v3.xlsx")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# run_demo.py (place in project root)
from src.params import DEFAULTS
from src.pricer import montecarlo_price, fair_coupon_for_target_yield
from src.reporter import save_results_to_excel, plot_payoff_hist
from src.simulator import correlated_default_sim
import numpy as np

p = DEFAULTS.copy()
price, std = montecarlo_price(p["pd_annual"], p["correlation"], p["tenor"], p["coupon_target"], \
    p["num_sims"], p["discount_rate"], p["recovery_rate"], face=1.0)
print(f"Monte Carlo PV price: {price:.6f}  STD: {std:.6f}")

fair_c = fair_coupon_for_target_yield(p["coupon_target"], p["pd_annual"], p["correlation"], \
    p["tenor"], p["num_sims"], p["discount_rate"], p["recovery_rate"], face=1.0)
print(f"Fair coupon (for target yield={p['coupon_target']:.2%}): {fair_c:.2%}")

# Save small report
#os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
#print("Created outputs folder:", os.path.exists(os.path.dirname(OUT_PATH)))
#print(f"Saving Excel to: {OUT_PATH}")
save_results_to_excel(OUT_PATH, p, price, std, fair_coupon=fair_c, extra="Example run")
#print("✅ Excel save function called successfully.")
# save_results_to_excel("outputs/example_report.xlsx", p, price, std, fair_coupon=fair_c, extra="Example run")
