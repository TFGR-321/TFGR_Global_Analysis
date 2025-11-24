#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 44-1 (revised):
Joint fit of TFGR parameters AND satellite-specific constant offsets.

Model:
  clk_bias_s = TFGR_dt(L; dt0, Lc, p, q) + b_sat

Key trick:
  For any TFGR params, best b_sat is analytical:
    b_sat = mean(clk_bias_s - TFGR_dt) per satellite
  So we only optimize 4 TFGR params, offsets solved inside.

Usage (1-line):
python Phase44_1_joint_sat_offset_fit.py --in_csv phase42_gps_only.csv --sat_col sat --L_col L_m --dt_col clk_bias_s --unit_m --out_prefix phase44_1b --plot
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def tfgr_dt(L, dt0, Lc, p, q):
    return dt0 * (1.0 + (L / Lc) ** p) ** q


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--sat_col", default="sat")
    ap.add_argument("--L_col", default="L_m")
    ap.add_argument("--dt_col", default="clk_bias_s")
    ap.add_argument("--unit_m", action="store_true")
    ap.add_argument("--unit_km", action="store_true")
    ap.add_argument("--out_prefix", default="phase44_1b")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)

    sat = df[args.sat_col].astype(str).values
    L = df[args.L_col].astype(float).values
    dt_obs = df[args.dt_col].astype(float).values

    if args.unit_km and not args.unit_m:
        L = L * 1000.0

    # Encode satellite ids 0..Nsat-1
    uniq_sats = sorted(np.unique(sat))
    sat_ids = {s: i for i, s in enumerate(uniq_sats)}
    sid = np.array([sat_ids[s] for s in sat])
    Nsat = len(uniq_sats)

    # Loss with analytic offsets
    def loss(theta):
        dt0, Lc, p, q = theta
        pred = tfgr_dt(L, dt0, Lc, p, q)
        resid = dt_obs - pred

        # analytic best offsets per sat
        offsets = np.zeros(Nsat)
        for i in range(Nsat):
            offsets[i] = resid[sid == i].mean()

        resid2 = resid - offsets[sid]
        return np.mean(resid2 ** 2)

    # Initial guess (Phase42C near values)
    theta0 = np.array([2.75e-5, 4.5e9, 0.19, 1.29])
    bounds = [(0, 1e-3), (1e7, 1e12), (0.01, 1.0), (0.1, 5.0)]

    res = minimize(loss, theta0, bounds=bounds, method="L-BFGS-B")
    dt0, Lc, p, q = res.x

    # Final offsets + residuals
    pred = tfgr_dt(L, dt0, Lc, p, q)
    resid = dt_obs - pred
    offsets = np.zeros(Nsat)
    for i in range(Nsat):
        offsets[i] = resid[sid == i].mean()
    resid2 = resid - offsets[sid]

    rms = np.sqrt(np.mean(resid2 ** 2))

    print("\n[Phase44-1b] Joint TFGR + sat-offset fit results:")
    print(f"  dt0 = {dt0:.6e} s")
    print(f"  Lc  = {Lc:.6e} m")
    print(f"  p   = {p:.6f}")
    print(f"  q   = {q:.6f}")
    print(f"  RMS (after offsets) = {rms:.6e} s")

    # Save offsets table
    out_offsets = pd.DataFrame({
        "sat": uniq_sats,
        "b_sat": offsets
    })
    out_csv = f"{args.out_prefix}_sat_offsets.csv"
    out_offsets.to_csv(out_csv, index=False)
    print(f"[Phase44-1b] Saved sat offsets -> {out_csv}")

    # Optional plot:
    # show (dt_obs - b_sat) vs L with TFGR curve
    if args.plot:
        dt_debiased = dt_obs - offsets[sid]
        order = np.argsort(L)

        plt.figure()
        plt.plot(L[order], dt_debiased[order], ".", ms=1.5, label="(clk_bias - b_sat)")
        plt.plot(L[order], pred[order], "-", lw=2, label="TFGR fit")
        plt.xlabel("L [m]")
        plt.ylabel("Δt debiased [s]")
        plt.legend()
        plt.tight_layout()
        png = f"{args.out_prefix}_joint_fit.png"
        plt.savefig(png, dpi=200)
        print(f"[Phase44-1b] Saved plot -> {png}")


if __name__ == "__main__":
    main()
