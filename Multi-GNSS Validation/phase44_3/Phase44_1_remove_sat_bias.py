#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 44-1: Remove satellite-specific clock bias (mean offset) and refit TFGR.

Usage example:
python Phase44_1_remove_sat_bias.py \
  --in_csv phase42_gps_only.csv \
  --out_csv phase44_bias_removed.csv \
  --sat_col sat \
  --L_col L_m \
  --dt_col dt_obs \
  --unit_m \
  --do_fit \
  --out_prefix phase44_1
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def tfgr_dt(L, dt0, Lc, p, q):
    """TFGR time correction model."""
    return dt0 * (1.0 + (L / Lc) ** p) ** q


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sat_col", default="sat")
    ap.add_argument("--L_col", default="L_m")
    ap.add_argument("--dt_col", default="dt_obs")
    ap.add_argument("--unit_m", action="store_true",
                    help="L is in meters (default).")
    ap.add_argument("--unit_km", action="store_true",
                    help="L is in kilometers; converted to meters.")
    ap.add_argument("--do_fit", action="store_true",
                    help="Run TFGR fit after bias removal.")
    ap.add_argument("--sr_subtract", action="store_true",
                    help="Assume SR correction already stored separately; no SR op here.")
    ap.add_argument("--out_prefix", default="phase44_1")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)

    # --- Load columns
    sat = df[args.sat_col].astype(str)
    L = df[args.L_col].astype(float).values
    dt = df[args.dt_col].astype(float).values

    # --- Unit normalize
    if args.unit_km and not args.unit_m:
        L = L * 1000.0

    # --- Satellite mean bias removal
    df["_sat"] = sat
    df["_dt"] = dt
    sat_means = df.groupby("_sat")["_dt"].mean()
    df["sat_bias_mean"] = df["_sat"].map(sat_means)
    df["dt_bias_removed"] = df["_dt"] - df["sat_bias_mean"]

    # --- Save
    out_cols = list(df.columns)
    df.to_csv(args.out_csv, index=False)
    print(f"[Phase44-1] Saved bias-removed CSV -> {args.out_csv}")
    print("[Phase44-1] Satellite mean biases (sec):")
    print(sat_means.head(20))

    # --- Optional TFGR fit
    if args.do_fit:
        y = df["dt_bias_removed"].values
        x = L

        # Initial guesses close to Phase42C
        p0 = [2.75e-5, 4.5e9, 0.19, 1.29]  # dt0, Lc, p, q
        bounds = (
            [0, 1e7, 0.01, 0.1],   # lower
            [1e-3, 1e12, 1.0, 5.0] # upper
        )

        popt, pcov = curve_fit(tfgr_dt, x, y, p0=p0, bounds=bounds, maxfev=200000)
        dt0, Lc, p, q = popt
        yhat = tfgr_dt(x, *popt)
        rms = np.sqrt(np.mean((y - yhat) ** 2))

        print("\n[Phase44-1] TFGR Fit after bias removal:")
        print(f"  dt0 = {dt0:.6e} s")
        print(f"  Lc  = {Lc:.6e} m")
        print(f"  p   = {p:.6f}")
        print(f"  q   = {q:.6f}")
        print(f"  RMS = {rms:.6e} s")

        # Plot
        if args.plot:
            order = np.argsort(x)
            plt.figure()
            plt.plot(x[order], y[order], ".", ms=2, label="bias removed")
            plt.plot(x[order], yhat[order], "-", lw=2, label="TFGR fit")
            plt.xlabel("L [m]")
            plt.ylabel("Δt_bias_removed [s]")
            plt.legend()
            plt.tight_layout()
            png = f"{args.out_prefix}_tfgr_fit_bias_removed.png"
            plt.savefig(png, dpi=200)
            print(f"[Phase44-1] Saved plot -> {png}")


if __name__ == "__main__":
    main()
