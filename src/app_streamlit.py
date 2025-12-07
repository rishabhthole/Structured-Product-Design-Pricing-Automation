# To run use the below in terminal
# streamlit run "/Users/rishabhthole/Desktop/Data/Projects/Structured Product Design & Pricing Automation/src/app_streamlit.py"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# src/app_streamlit.py
import streamlit as st
import pandas as pd
from src.pricer import montecarlo_price, fair_coupon_for_target_yield

st.title("CLN Simulator")

# User inputs
names = st.text_input("Names (comma separated)", "JPM,DB,HSBC").split(",")
pd_str = st.text_input("Annual PDs (comma)", "0.02,0.03,0.015")
pd_annual = [float(x.strip()) for x in pd_str.split(",")]
corr = st.slider("Correlation", 0.0, 0.99, 0.35)
tenor = st.number_input("Tenor (years)", min_value=1, max_value=10, value=2)
coupon = st.number_input("Coupon (annual)", min_value=0.0, value=0.08, step=0.001)
num_sims = st.number_input("Num sims", min_value=1000, max_value=500000, value=50000)

if st.button("Run Monte Carlo"):
    price, std = montecarlo_price(pd_annual, corr, tenor, coupon, int(num_sims), discount_rate=0.03, recovery=0.4)
    st.write(f"Estimated PV price: {price:.4f}  (std {std:.4f})")
    # Optionally compute fair coupon for a target yield:
    target_yield = st.number_input("Target yield to calibrate fair coupon", value=0.08)
    if st.button("Calibrate fair coupon"):
        fair = fair_coupon_for_target_yield(target_yield, pd_annual, corr, tenor, int(num_sims), discount_rate=0.03, recovery=0.4)
        st.write(f"Fair coupon (annual): {fair:.4%}")
