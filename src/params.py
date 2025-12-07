# src/params.py
DEFAULTS = {
    "names": ["JPM", "DB", "HSBC"],
    # annual default probabilities (annual PD) per name
    "pd_annual": [0.02, 0.03, 0.015],
    "correlation": 0.35,    # equal pairwise correlation
    "coupon_target": 0.08,  # target coupon (annual)
    "tenor": 2,             # years
    "face": 1_000_000,      # notional per note
    "num_sims": 200_000,
    "discount_rate": 0.03,  # flat yield curve for discounting
    "recovery_rate": 0.4,   # assumed recovery on default
}
