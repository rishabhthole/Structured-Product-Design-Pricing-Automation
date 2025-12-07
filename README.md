Automated Pricing & Payoff Simulator for Credit-Linked Structured Notes (CLN-Structuring)
📄 Overview

This project models, prices, and analyses Credit-Linked Structured Notes (CLNs) — fixed-income instruments whose returns depend on the credit performance of an underlying bond basket.

It provides:

Monte-Carlo-based correlated default simulation

Fair-coupon calibration (price = par)

Payoff distribution & sensitivity dashboards

Excel / Streamlit automation

The goal is to replicate the workflow of a Fixed Income Structuring Analyst: combining quantitative modeling, product design, and automation to evaluate yield–risk trade-offs.

This simulator allows you to:

Quantify expected loss and fair coupon levels

Analyse sensitivities to correlation, PD, and recovery assumptions

Generate client-ready payoff and sensitivity charts

🧩 Project Structure

cln-structuring/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sample_basket.csv
├─ src/
│  ├─ __init__.py
│  ├─ params.py          # default parameter set
│  ├─ utils.py           # helpers for PV, discounting, correlations
│  ├─ simulator.py       # correlated default Monte-Carlo engine
│  ├─ pricer.py          # fair-coupon calibration & pricing
│  ├─ reporter.py        # Excel export & visualization
│  └─ app_streamlit.py   # optional Streamlit dashboard
├─ notebooks/
│  └─ demo.ipynb         # demonstration notebook
└─ outputs/
   └─ example_report.xlsx


⚙️ Installation

# Clone the repo
git clone https://github.com/<your-username>/cln-structuring.git
cd cln-structuring

# Install dependencies
pip install -r requirements.txt

🧮 Quick Start

Run the example notebook:

jupyter notebook notebooks/demo.ipynb

Or execute a full simulation from command line:
python src/run_demo.py

Example terminal output:
Fair annual coupon: 0.0812
Monte Carlo PV mean: 1.0001
Plots and summary exported to /outputs


📊 Sample Outputs
Chart	Description

	Distribution of simulated present values

	Cumulative distribution (probability that PV < x)

(add screenshots from your /outputs folder once you run it)

🧱 Methodology

Correlated default simulation

Gaussian-copula model to simulate correlated credit events across names.

Cash-flow modeling

Quarterly coupon flows stop upon default.

Recovery applied on defaulted notional.

Discounting

Discounted using interpolated zero-rate curve.

Fair-coupon calibration

Iteratively solves for coupon such that PV(note) = par.

Sensitivity analysis

Correlation, PD, and recovery shocks.

🧰 Tech Stack

Python 3.11+

numpy, pandas, scipy, matplotlib

openpyxl for Excel export

streamlit (optional) for dashboard UI

jupyter for research & visualization

🚀 Example Use-Cases

Evaluate credit-linked or rate-linked structured notes

Assess yield vs. correlation sensitivity for portfolio credit risk

Demonstrate applied quantitative finance & structuring skills in interviews

📈 Future Enhancements

Add t-Copula for tail-dependency modeling

Integrate real yield curves via Bloomberg or FRED API

Extend to hybrid notes (credit + rates + FX)

Include VaR / Expected Shortfall on payoff distribution
