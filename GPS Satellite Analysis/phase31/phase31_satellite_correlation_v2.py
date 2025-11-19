#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
phase31_satellite_correlation_v2.py
TFGR Multi-Satellite Correlation (CSV column auto-match version)

Your CSV columns:
    time, sat, elev_deg, az_deg,
    slant_total_m, expected_slant_m,
    residual_m, station,
    range_m, phi_t, fit_residual

We use:
    sat        → satellite ID
    residual_m → residual (y)
    phi_t      → TFGR Φ_t(L)
    range_m    → L (optional)
    station    → station label
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear_fit_with_stats(y, x):
    """Fit y = A x + b (linear regression)"""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = ~np.isnan(y) & ~np.isnan(x)
    y = y[mask]
    x = x[mask]

    N = len(y)
    if N < 3:
        return (np.nan,)*8

    RSS_before = np.sum(y ** 2)
    RMS_before = math.sqrt(RSS_before / N)

    A, b = np.polyfit(x, y, 1)
    y_pred = A * x + b
    residuals = y - y_pred

    RSS_after = np.sum(residuals ** 2)
    RMS_after = math.sqrt(RSS_after / N)

    # AIC
    if RSS_before <= 0: RSS_before = 1e-30
    if RSS_after <= 0: RSS_after = 1e-30

    AIC_before = N * math.log(RSS_before / N) + 2 * 1
    AIC_after = N * math.log(RSS_after / N) + 2 * 2
    delta_AIC = AIC_before - AIC_after

    TSS = np.sum((y - np.mean(y)) ** 2)
    R2 = np.nan if TSS == 0 else 1 - RSS_after / TSS

    return A, b, RMS_before, RMS_after, AIC_before, AIC_after, delta_AIC, R2


def analyse_csv(csv_path, out_prefix, out_dir, plot=False):

    df = pd.read_csv(csv_path)

    required = ["sat", "residual_m", "phi_t", "station"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    sats = sorted(df["sat"].dropna().unique().tolist())
    rows = []

    for sat in sats:
        sub = df[df["sat"] == sat]
        N_points = len(sub)

        if N_points < 3:
            rows.append({
                "station": sub["station"].iloc[0] if N_points > 0 else "unknown",
                "sat": sat,
                "N_points": N_points,
                "A_sat": np.nan,
                "b_sat": np.nan,
                "RMS_before": np.nan,
                "RMS_after": np.nan,
                "delta_RMS": np.nan,
                "AIC_before": np.nan,
                "AIC_after": np.nan,
                "delta_AIC": np.nan,
                "R2": np.nan,
            })
            continue

        y = sub["residual_m"].values
        phi = sub["phi_t"].values
        station_label = sub["station"].iloc[0]

        A, b, RMS_before, RMS_after, AIC_before, AIC_after, delta_AIC, R2 = \
            linear_fit_with_stats(y, phi)

        rows.append({
            "station": station_label,
            "sat": sat,
            "N_points": N_points,
            "A_sat": A,
            "b_sat": b,
            "RMS_before": RMS_before,
            "RMS_after": RMS_after,
            "delta_RMS": RMS_after - RMS_before,
            "AIC_before": AIC_before,
            "AIC_after": AIC_after,
            "delta_AIC": delta_AIC,
            "R2": R2,
        })

        if plot and not np.isnan(A):
            plt.figure()
            plt.scatter(phi, y, s=10, alpha=0.7)
            x_line = np.linspace(np.nanmin(phi), np.nanmax(phi), 200)
            y_line = A * x_line + b

            plt.plot(x_line, y_line)
            plt.xlabel("phi_t")
            plt.ylabel("residual_m")
            plt.title(f"{station_label} : {sat}")
            plt.grid()

            out_img = f"{out_prefix}_{station_label}_{sat}.png"
            plt.savefig(os.path.join(out_dir, out_img), dpi=200)
            plt.close()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Input CSV file (residual data)")
    parser.add_argument("--out_prefix", default="phase31",
                        help="Prefix for output files")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory")
    parser.add_argument("--plot", action="store_true",
                        help="Generate fit plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_summary = analyse_csv(
        csv_path=args.csv,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir,
        plot=args.plot,
    )

    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv")
    df_summary.to_csv(out_csv, index=False)

    print(f"[INFO] Saved summary → {out_csv}")


if __name__ == "__main__":
    main()
