#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----------------------------
# TFGR model
# ----------------------------
def tfgr_model(L, dt0, Lc, p, q):
    return dt0 * (1.0 + (L / Lc)**p)**q

# ----------------------------
# Fit function
# ----------------------------
def fit_tfgr(df):
    L = df["L_m"].values.astype(float)
    dt = df["clk_bias_s"].values.astype(float)

    # initial guesses
    # ---- safe initial values ----
    dt0_init = max(1e-8, abs(np.median(dt)))   # ~1e-5
    Lc_init  = 4e9
    p_init   = 0.2
    q_init   = 1.3

    p0 = [dt0_init, Lc_init, p_init, q_init]

# ---- safe wide bounds ----
    bounds = (
        [1e-12,   1e6,   0.0,   0.0],
        [1e-2,    1e12,  5.0,   5.0]
)

    popt, pcov = curve_fit(
        tfgr_model, L, dt,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
        ftol=1e-15,
        xtol=1e-15
)



    dt0_hat, Lc_hat, p_hat, q_hat = popt
    dt_fit = tfgr_model(L, *popt)
    residual = dt - dt_fit
    rms = np.sqrt(np.mean(residual**2))

    return popt, rms, dt_fit, residual

# ----------------------------
# Plot helper
# ----------------------------
def plot_results(df, dt_fit, residual, out_prefix):
    L = df["L_m"]
    dt = df["clk_bias_s"]

    # dt vs L
    plt.figure(figsize=(8,6))
    plt.scatter(L, dt, s=8, label="Observed", alpha=0.6)
    plt.plot(L, dt_fit, "r-", label="TFGR Fit", linewidth=2)
    plt.xlabel("L [m]")
    plt.ylabel("Clock bias dt [s]")
    plt.title("TFGR Fit (GPS only)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_dt_vs_L.png", dpi=300)
    plt.close()

    # residual vs L
    plt.figure(figsize=(8,6))
    plt.scatter(L, residual, s=8, alpha=0.6)
    plt.xlabel("L [m]")
    plt.ylabel("Residual [s]")
    plt.title("Residual (dt_obs - dt_fit)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residual_vs_L.png", dpi=300)
    plt.close()


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="TFGR fit for GPS-only merged orbit-clock file")
    ap.add_argument("--csv", required=True, help="Input GPS-only merged CSV (phase42_gps_only.csv)")
    ap.add_argument("--out", required=True, help="Output prefix")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    print(f"[loaded] rows={len(df)} sats={df['sat'].nunique()}")

    popt, rms, dt_fit, residual = fit_tfgr(df)

    dt0_hat, Lc_hat, p_hat, q_hat = popt

    # Save results text
    txt = (
        f"TFGR Fit Results (GPS only)\n"
        f"=============================\n"
        f"dt0_hat_s = {dt0_hat:.12e}\n"
        f"Lc_hat_m  = {Lc_hat:.6e}\n"
        f"p_hat     = {p_hat:.6f}\n"
        f"q_hat     = {q_hat:.6f}\n"
        f"RMS_res_s = {rms:.12e}\n"
        f"rows      = {len(df)}\n"
    )

    with open(f"{args.out}_fit.txt", "w") as f:
        f.write(txt)

    print(txt)

    # Save residual CSV
    out_df = df.copy()
    out_df["dt_fit"] = dt_fit
    out_df["residual"] = residual
    out_df.to_csv(f"{args.out}_fit.csv", index=False)

    # Plots
    plot_results(df, dt_fit, residual, args.out)
    print(f"[saved] {args.out}_dt_vs_L.png")
    print(f"[saved] {args.out}_residual_vs_L.png")
    print(f"[saved] {args.out}_fit.csv")
    print(f"[saved] {args.out}_fit.txt")

if __name__ == "__main__":
    main()
