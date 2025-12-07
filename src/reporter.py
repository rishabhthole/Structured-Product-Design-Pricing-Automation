# src/reporter.py
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import io

def save_results_to_excel(filename, params, price, std, fair_coupon=None, extra=None):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Parameter", "Value"])
    for k, v in params.items():
        ws.append([k, str(v)])
    ws.append([])
    ws.append(["Monte Carlo Price (PV)", price])
    ws.append(["Price STD", std])
    if fair_coupon is not None:
        ws.append(["Fair Coupon (annual)", fair_coupon])
    if extra:
        ws.append([])
        ws.append(["Notes"])
        ws.append([extra])
    wb.save(filename)

def plot_payoff_hist(payoffs, outpath="payoff_hist.png"):
    plt.figure()
    plt.hist(payoffs, bins=80)
    plt.title("Final Payoff Distribution")
    plt.xlabel("Final Value (PV at t=0)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
