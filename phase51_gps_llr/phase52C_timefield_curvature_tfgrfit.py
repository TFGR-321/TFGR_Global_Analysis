#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 52C: Time-field curvature with TFGR model fit (robust version)
---------------------------------------------------
GPS + LLR データを TFGR 型 Δt(L) = A([1+(L/Lc)^p]^q − 1) にフィット。
Φ_t の勾配と曲率を広範囲スキャンで解析し、
臨界スケール Lc⊕ ≈ 4×10⁹ m の位置を推定。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# -------------------------------------------------
# データ読込関数（数値列のみを選別）
# -------------------------------------------------
def load_dataset(path, L_col=None, dt_col=None):
    df = pd.read_csv(path)

    # 数値列のみを抽出
    num_df = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        raise ValueError(f"⚠️ {path} に有効な数値列が見つかりません。")

    # L列候補
    if L_col and L_col in num_df.columns:
        L = num_df[L_col].values
    else:
        L = num_df.iloc[:, 0].values

    # Δt列候補
    if dt_col and dt_col in num_df.columns:
        dt = num_df[dt_col].values
    else:
        dt = num_df.iloc[:, 1].values if num_df.shape[1] > 1 else num_df.iloc[:, 0].values

    # ソート
    order = np.argsort(L)
    return L[order].astype(float), dt[order].astype(float)


# -------------------------------------------------
# TFGRモデル関数
# -------------------------------------------------
def tfgr_dt(L, A, Lc, p, q):
    return A * ((1.0 + (L / Lc) ** p) ** q - 1.0)


def compute_derivatives(L, y):
    y = np.asarray(y)
    L = np.asarray(L)
    dy_dL = np.gradient(y, L)
    d2y_dL2 = np.gradient(dy_dL, L)
    return dy_dL, d2y_dL2


def detect_zero_crossing(L, f):
    f = np.asarray(f)
    L = np.asarray(L)
    idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    x0, x1 = L[i], L[i + 1]
    y0, y1 = f[i], f[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


# -------------------------------------------------
# メイン関数
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 52C robust: TFGR fit + extended curvature scan"
    )
    parser.add_argument("--gps_csv", required=True)
    parser.add_argument("--llr_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--L_col", default=None)
    parser.add_argument("--dt_col", default=None)
    parser.add_argument("--p", type=float, default=0.21)
    parser.add_argument("--q", type=float, default=1.32)
    parser.add_argument("--scan_points", type=int, default=1500)
    args = parser.parse_args()

    # ---- データ読込 ----
    L_gps, dt_gps = load_dataset(args.gps_csv, args.L_col, args.dt_col)

    # LLRをまとめて平均化
    llr_paths = [p.strip() for p in args.llr_csv.split(",") if p.strip()]
    L_all_llr, dt_all_llr = [], []
    for path in llr_paths:
        L, dt = load_dataset(path, args.L_col, args.dt_col)
        L_all_llr.append(L)
        dt_all_llr.append(dt)

    # 共通logスケールで補間
    all_L = np.concatenate(L_all_llr)
    L_common = np.logspace(np.log10(all_L.min()), np.log10(all_L.max()), 200)
    dt_interp_list = []
    for L, dt in zip(L_all_llr, dt_all_llr):
        f = interp1d(L, dt, kind="linear", fill_value="extrapolate")
        dt_interp_list.append(f(L_common))
    dt_llr_mean = np.mean(np.vstack(dt_interp_list), axis=0)

    # 結合
    L_all = np.concatenate([L_gps, L_common])
    dt_all = np.concatenate([dt_gps, dt_llr_mean])

    # ---- TFGRフィット ----
    p_fix, q_fix = args.p, args.q

    def model(L, A, Lc):
        return tfgr_dt(L, A, Lc, p_fix, q_fix)

    L_min, L_max = float(L_all.min()), float(L_all.max())
    Lc0 = np.exp((np.log(L_min) + np.log(L_max)) / 2)
    A0 = (dt_all.max() - dt_all.min()) / ((1 + (L_max / Lc0) ** p_fix) ** q_fix - 1)
    popt, _ = curve_fit(model, L_all, dt_all, p0=[A0, Lc0], maxfev=20000)
    A_hat, Lc_hat = popt

    # ---- 広域スキャン ----
    L_scan = np.logspace(np.log10(L_min / 100), np.log10(L_max * 1e3), args.scan_points)
    dt_model = model(L_scan, A_hat, Lc_hat)
    dphi_dL, d2phi_dL2 = compute_derivatives(L_scan, dt_model)
    L_zero = detect_zero_crossing(L_scan, d2phi_dL2)

    # ---- プロット ----
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(L_scan, dt_model)
    ax[0].set_xscale("log")
    ax[0].set_ylabel("Δt(L)")
    ax[0].set_title("TFGR Fit & Time-field Curvature (Extended)")
    ax[0].grid(True)

    ax[1].plot(L_scan, dphi_dL, color="blue")
    ax[1].set_xscale("log")
    ax[1].set_ylabel("∂Φₜ/∂L")
    ax[1].grid(True)

    ax[2].plot(L_scan, d2phi_dL2, color="green")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("L [m]")
    ax[2].set_ylabel("∂²Φₜ/∂L²")
    ax[2].grid(True)
    if L_zero:
        ax[2].axvline(L_zero, linestyle="--", label=f"L_zero ≈ {L_zero:.2e} m")
    ax[2].axvline(Lc_hat, color="red", linestyle=":", label=f"Lc_fit ≈ {Lc_hat:.2e} m")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(f"{args.out}_tfgr_curvature_scan.png", dpi=200)

    # ---- 出力 ----
    summary = {
        "A_hat": A_hat,
        "Lc_hat_m": Lc_hat,
        "L_zero_curvature_m": L_zero,
        "L_min": L_min,
        "L_max": L_max,
    }
    pd.DataFrame([summary]).to_csv(f"{args.out}_tfgr_fit_summary.csv", index=False)
    print(f"TFGR fit Lc ≈ {Lc_hat:.3e} m, zero-curvature ≈ {L_zero:.3e} m" if L_zero else f"TFGR fit Lc ≈ {Lc_hat:.3e} m")


if __name__ == "__main__":
    main()
