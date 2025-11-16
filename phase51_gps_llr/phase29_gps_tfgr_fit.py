#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
phase29_gps_tfgr_fit.py
-----------------------
TFGR（時間場）補正項をGPS衛星の時計残差に導入してフィットし、
GR補正後に残る微小残差のRMS低下や情報量基準(AIC)改善を評価します。

■ 入力CSVの想定カラム（最低限）
- time_utc:  ISO8601 or "YYYY-mm-dd HH:MM:SS"
- sat:       衛星ID（任意）
- elev_deg:  衛星仰角 [deg]（無ければNaN可）
- residual_s: GR補正まで反映後の時計残差 [sec]
- (optional) sigma_s: 残差の標準偏差 [sec]
- (optional) range_m: 幾何距離 [m]

■ パラメータ
- Lc, p, q:  S-DTFTの経験式パラメータ（Phase 16推定）
- Lmode:     'range'|'elev'|'fixed' でLを推定
- fit_mode:  'A' or 'alpha' （r' = r - A*Phi_t(L)）

■ 出力
- <out>_results.txt
- <out>_with_model.csv
- <out>_timeseries.png, <out>_scatter_fit.png, <out>_hist.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C = 299792458.0    # [m/s]
RE = 6371000.0     # [m]
GPS_ALT = 20200000.0  # [m]

def approx_slant_range_from_elev(elev_deg, re=RE, h=GPS_ALT):
    elev_rad = np.radians(np.clip(elev_deg, 0.0, 89.9))
    psi = (np.pi/2.0) - elev_rad
    L = np.sqrt( (re+h)**2 + re**2 - 2.0*re*(re+h)*np.cos(psi) )
    return L

def compute_phi_t(L, Lc, p, q):
    x = (L / Lc)**p
    return (1.0 + x)**q - 1.0

def fit_A_least_squares(phi, r, w=None):
    if w is None:
        denom = np.dot(phi, phi)
        if denom <= 0:
            return 0.0
        return np.dot(phi, r) / denom
    denom = np.dot(phi*w, phi)
    if denom <= 0:
        return 0.0
    return np.dot(phi*w, r) / denom

def AIC(n, rss, k):
    if n<=0 or rss<=0: return np.inf
    return n*np.log(rss/n) + 2*k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Lc", type=float, default=4.0e9)
    ap.add_argument("--p", type=float, default=0.21)
    ap.add_argument("--q", type=float, default=1.32)
    ap.add_argument("--Lmode", choices=["range","elev","fixed"], default="elev")
    ap.add_argument("--Lfixed", type=float, default=2.7e7)
    ap.add_argument("--fit_mode", choices=["A","alpha"], default="A")
    ap.add_argument("--time_col", default="time_utc")
    ap.add_argument("--resid_col", default="residual_s")
    ap.add_argument("--sigma_col", default="sigma_s")
    ap.add_argument("--elev_col", default="elev_deg")
    ap.add_argument("--range_col", default="range_m")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # L estimate
    if args.Lmode == "range" and (args.range_col in df.columns):
        L = df[args.range_col].to_numpy(np.float64)
    elif args.Lmode == "elev" and (args.elev_col in df.columns):
        L = approx_slant_range_from_elev(df[args.elev_col].to_numpy(np.float64))
    else:
        L = np.full(len(df), float(args.Lfixed), dtype=np.float64)

    phi = compute_phi_t(L, args.Lc, args.p, args.q)

    if args.resid_col not in df.columns:
        raise ValueError(f"Residual column '{args.resid_col}' not found.")
    r = df[args.resid_col].to_numpy(np.float64)

    w = None
    if args.sigma_col in df.columns:
        sigma = df[args.sigma_col].to_numpy(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0/np.where(sigma>0, sigma**2, np.nan)
            w = np.where(np.isfinite(w), w, 0.0)

    # Fit
    A = fit_A_least_squares(phi, r, w=w)
    if args.fit_mode == "A":
        model = A * phi
        param_name, param_value = "A", A
    else:
        alpha = A * (C**2)
        model = A * phi
        param_name, param_value = "alpha", alpha

    resid_after = r - model

    # Metrics
    n = np.isfinite(r).sum()
    rss0 = np.nansum((r - np.nanmean(r))**2)
    rss1 = np.nansum((resid_after - np.nanmean(resid_after))**2)
    rms0 = np.sqrt(np.nanmean((r - np.nanmean(r))**2))
    rms1 = np.sqrt(np.nanmean((resid_after - np.nanmean(resid_after))**2))
    aic0 = AIC(n, rss0, 1)  # mean only
    aic1 = AIC(n, rss1, 2)  # mean + 1 coeff

    # Save CSV
    out_csv = f"{args.out}_with_model.csv"
    df_out = df.copy()
    df_out["L_m"] = L
    df_out["phi_t"] = phi
    df_out["model_tfgr_s"] = model
    df_out["residual_after_s"] = resid_after
    df_out.to_csv(out_csv, index=False)

    # Save results txt
    out_txt = f"{args.out}_results.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"TFGR fit\nCSV: {args.csv}\n")
        f.write(f"Params: Lc={args.Lc:.3e}, p={args.p:.4f}, q={args.q:.4f}, Lmode={args.Lmode}\n")
        f.write(f"Fitted {param_name} = {param_value:.6e}\n")
        f.write(f"RMS before: {rms0:.6e} s\n")
        f.write(f"RMS after : {rms1:.6e} s\n")
        f.write(f"AIC before: {aic0:.3f}\n")
        f.write(f"AIC after : {aic1:.3f}\n")
        f.write(f"ΔAIC: {aic1-aic0:.3f}\n")

    # Fig 1: timeseries by index (robust even without time)
    idx = np.arange(len(r))
    fig1 = plt.figure()
    plt.plot(idx, r, marker="o", linestyle="", label="before (post-GR)")
    plt.plot(idx, resid_after, marker=".", linestyle="", label="after TFGR")
    plt.xlabel("Index")
    plt.ylabel("Clock residual (s)")
    plt.legend()
    plt.title("GPS clock residuals: before vs after TFGR")
    fig1.savefig(f"{args.out}_timeseries.png", dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # Fig 2: scatter r vs phi
    fig2 = plt.figure()
    plt.plot(phi, r, marker="o", linestyle="", label="data")
    xline = np.linspace(np.nanmin(phi), np.nanmax(phi), 200)
    yline = A * xline
    plt.plot(xline, yline, linestyle="-", label="fit")
    plt.xlabel("Phi_t(L)")
    plt.ylabel("Clock residual (s)")
    plt.legend()
    plt.title("Residual vs Phi_t(L)")
    fig2.savefig(f"{args.out}_scatter_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # Fig 3: hist
    fig3 = plt.figure()
    plt.hist(r, bins=40, alpha=0.6, label="before (post-GR)")
    plt.hist(resid_after, bins=40, alpha=0.6, label="after TFGR")
    plt.xlabel("Clock residual (s)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Histogram of residuals: before vs after TFGR")
    fig3.savefig(f"{args.out}_hist.png", dpi=160, bbox_inches="tight")
    plt.close(fig3)

    print(f"[OK] Wrote: {out_csv}, {out_txt}")
    print(f"[OK] Figures: {args.out}_timeseries.png, {args.out}_scatter_fit.png, {args.out}_hist.png")

if __name__ == "__main__":
    main()
