# To run use the below in terminal (note the quotes if you have spaces in folder name)
# streamlit run "/Users/rishabhthole/Desktop/Data/Projects/Structured Product Design & Pricing Automation/src/app_streamlit_v3.py"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd

# Import functions and defaults from your project
from src.params import DEFAULTS
from src.pricer_v2 import montecarlo_price, fair_coupon_for_par
# fair coupon function name may differ between versions; try both
try:
    from src.pricer_v2 import fair_coupon_for_par as fair_coupon_solver
except Exception:
    try:
        from src.pricer_v2 import fair_coupon_for_target_yield as fair_coupon_solver
    except Exception:
        fair_coupon_solver = None  # we'll handle this later

# --------------------
st.set_page_config(page_title="CLN Simulator", layout="centered")
st.title("💰 CLN Simulator")

st.markdown("Monte Carlo pricing and fair-coupon calibration for a multi-name Credit-Linked Note (CLN).")

# Sidebar inputs
st.sidebar.header("Input parameters")
names = st.sidebar.text_input("Names (comma separated)", "JPM,DB,HSBC").split(",")
pd_str = st.sidebar.text_input("Annual PDs (comma separated)", "0.02,0.03,0.015")
pd_annual = [float(x.strip()) for x in pd_str.split(",") if x.strip()]

corr = st.sidebar.slider("Correlation", min_value=0.0, max_value=0.99, value=float(DEFAULTS.get("correlation", 0.35)), step=0.01)
tenor = st.sidebar.number_input("Tenor (years)", min_value=1, max_value=10, value=int(DEFAULTS.get("tenor", 2)))
coupon = st.sidebar.number_input("Coupon (annual)", min_value=0.0, value=0.08, step=0.001)
num_sims = st.sidebar.number_input("Number of simulations", min_value=1000, max_value=500000, value=int(DEFAULTS.get("num_sims", 50000)), step=1000)

# Use discount curve defaults from params (project-wide)
discount_curve_maturities = DEFAULTS.get("discount_curve_maturities", [0.25, 0.5, 1, 2, 5, 10])
discount_curve_rates = DEFAULTS.get("discount_curve_rates", [0.005, 0.007, 0.01, 0.02, 0.03, 0.035])
periods_per_year = DEFAULTS.get("periods_per_year", 4)
recovery = st.sidebar.number_input("Recovery rate", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

# Session state
if "mc_price" not in st.session_state:
    st.session_state.mc_price = None
    st.session_state.mc_std = None
    st.session_state.fair_coupon = None
    st.session_state.last_inputs = None

# Helper to compare params (so we don't reuse incompatible cached result)
def inputs_key():
    return dict(names=names, pd_annual=tuple(pd_annual), corr=float(corr), tenor=int(tenor), coupon=float(coupon),
                num_sims=int(num_sims), recovery=float(recovery))

# Run Monte Carlo (uses the correct signature for montecarlo_price in your project)
st.subheader("Monte Carlo pricing")
if st.button("Run Monte Carlo"):
    st.session_state.last_inputs = inputs_key()
    with st.spinner("Running Monte Carlo..."):
        # Project's montecarlo_price signature in your repo expects:
        # montecarlo_price(pd_annual, correlation, tenor, coupon, num_sims,
        #                  discount_curve_maturities, discount_curve_rates,
        #                  recovery=..., periods_per_year=..., rng=None)
        pv_vals, default_time = montecarlo_price(
            pd_annual,
            float(corr),
            int(tenor),
            float(coupon),
            int(num_sims),
            discount_curve_maturities,
            discount_curve_rates,
            recovery,
            periods_per_year,
            rng=None
        )
        st.session_state.mc_price = pv_vals.mean()
        st.session_state.mc_std = pv_vals.std()
    st.success(f"Estimated PV price: {st.session_state.mc_price:.6f}  (std {st.session_state.mc_std:.6f})")

# Show existing Monte Carlo result if present
if st.session_state.mc_price is not None:
    st.markdown(f"**Last simulation PV:** {st.session_state.mc_price:.6f}  |  **STD:** {st.session_state.mc_std:.6f}")

    # Fair coupon calibration (only if solver available)
    st.subheader("Fair coupon calibration")
    target_yield = st.number_input("Target yield (annual, to calibrate fair coupon)", value=0.08, step=0.001)

    if fair_coupon_solver is None:
        st.warning("Fair-coupon solver function not found in src.pricer. Please ensure `fair_coupon_for_par` or `fair_coupon_for_target_yield` exists.")
    else:
        if st.button("Calibrate Fair Coupon"):
            with st.spinner("Calibrating fair coupon..."):
                fair_c = fair_coupon_for_par(
                    pd_annual,
                    float(corr),
                    int(tenor),
                    int(num_sims),
                    discount_rate=0.03,  # pass as keyword argument
                    recovery=recovery
                )
            st.session_state.fair_coupon = fair_c
            st.success(f"Calibrated fair coupon (annual): {fair_c:.4%}")
            #     except TypeError:
            #         try:
            #             # try signature: fair_coupon_for_target_yield(target_yield, pd_annual, corr, tenor, num_sims, discount_rate, recovery)
            #             fair_c = fair_coupon_solver(
            #                 target_yield,
            #                 pd_annual,
            #                 float(corr),
            #                 int(tenor),
            #                 int(num_sims),
            #                 discount_curve_rates[0],  # fallback single discount rate
            #                 recovery
            #             )
            #         except Exception as e:
            #             st.error(f"Could not call fair coupon solver automatically: {e}")
            #             fair_c = None

            # if fair_c is not None:
            #     st.session_state.fair_coupon = fair_c
            #     st.success(f"Calibrated fair coupon (annual): {fair_c:.4%}")

# Summary table
if st.session_state.mc_price is not None or st.session_state.fair_coupon is not None:
    st.markdown("---")
    st.subheader("Summary")
    summary = {
        "Names": ", ".join(names),
        "Tenor": tenor,
        "Correlation": corr,
        "Coupon (input)": coupon,
        "Sim PV": st.session_state.mc_price,
        "Sim STD": st.session_state.mc_std,
        "Fair coupon": st.session_state.fair_coupon
    }
    st.table(pd.DataFrame([summary]))

st.caption("Built by you — CLN simulator")
