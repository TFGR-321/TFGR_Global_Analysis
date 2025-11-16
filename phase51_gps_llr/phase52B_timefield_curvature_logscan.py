#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 52B: Time-field curvature log-scan interpolation
-----------------------------------------------------
目的：
 GPS（地球表層）と LLR（月面反射）間のスケール領域（10³〜10⁹ m）を
 対数スケールで補間し、Φₜ の勾配および曲率を連続的に解析する。

出力：
 - phase52B_xxx_curvature_logscan.png
 - phase52B_xxx_logscan_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# -------------------------------------------------
# ユーティリティ関数
# -------------------------------------------------
def load_dataset(path, L_col=None, dt_col=None):
    """CSVからLとΔt列を読み込み"""
    df = pd.read_csv(path)
    if L_col is None:
        L_col = [c for c in df.columns if "L" in c or "distance" in c.lower()][0]
    if dt_col is None:
        dt_col = [c for c in df.columns if "dt" in c.lower()][0]
    df = df[[L_col, dt_col]].dropna()
    df = df.sort_values(L_col)
    return df[L_col].values, df[dt_col].values


def compute_phi(L, dt, c=299792458.0, dt0=None):
    """Φₜ = c² Δt / Δt₀ を計算"""
    if dt0 is None:
        dt0 = np.median(np.abs(dt))
        if dt0 == 0:
            dt0 = 1.0
    return (c ** 2) * dt / dt0


def logspace_interpolation(L, phi, n_points=800):
    """L–Φₜ関係を対数スケール補間"""
    Lmin, Lmax = L.min(), L.max()
    logL = np.logspace(np.log10(Lmin), np.log10(Lmax), n_points)
    interp_func = interp1d(L, phi, kind="linear", fill_value="extrapolate")
    phi_interp = interp_func(logL)
    return logL, phi_interp


def compute_derivatives(L, phi):
    """Φₜ の一次・二次微分"""
    dphi_dL = np.gradient(phi, L)
    d2phi_dL2 = np.gradient(dphi_dL, L)
    return dphi_dL, d2phi_dL2


def detect_zero_curvature(L, d2phi_dL2):
    """∂²Φₜ/∂L² = 0 の位置を線形補間で検出"""
    sign_change = np.where(np.sign(d2phi_dL2[:-1]) * np.sign(d2phi_dL2[1:]) < 0)[0]
    if len(sign_change) == 0:
        return None
    idx = sign_change[0]
    x0, x1 = L[idx], L[idx + 1]
    y0, y1 = d2phi_dL2[idx], d2phi_dL2[idx + 1]
    L_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
    return L_zero


# -------------------------------------------------
# メイン解析関数
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 52B: Time-field curvature log-scan interpolation."
    )
    parser.add_argument("--gps_csv", required=True)
    parser.add_argument("--llr_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--L_col", default=None)
    parser.add_argument("--dt_col", default=None)
    parser.add_argument("--c", type=float, default=299792458.0)
    args = parser.parse_args()

    # --- GPSデータ読み込み ---
    L_gps, dt_gps = load_dataset(args.gps_csv, args.L_col, args.dt_col)

    # --- LLRデータ読み込みと補間平均 ---
    LLR_paths = [p.strip() for p in args.llr_csv.split(",") if p.strip()]
    LLR_data = [load_dataset(p, args.L_col, args.dt_col) for p in LLR_paths]

    all_L = np.concatenate([arr[0] for arr in LLR_data])
    L_common = np.logspace(np.log10(all_L.min()), np.log10(all_L.max()), 200)

    dt_interp_list = []
    for L, dt in LLR_data:
        f = interp1d(L, dt, kind="linear", fill_value="extrapolate")
        dt_interp_list.append(f(L_common))
    dt_llr_mean = np.mean(np.vstack(dt_interp_list), axis=0)

    L_llr = L_common
    dt_llr = dt_llr_mean

    # --- GPSとLLRを統合 ---
    L_combined = np.concatenate([L_gps, L_llr])
    dt_combined = np.concatenate([dt_gps, dt_llr])

    phi = compute_phi(L_combined, dt_combined, c=args.c)
    L_log, phi_interp = logspace_interpolation(L_combined, phi, n_points=1000)
    dphi_dL, d2phi_dL2 = compute_derivatives(L_log, phi_interp)

    # --- 曲率ゼロ点検出 ---
    L_zero = detect_zero_curvature(L_log, d2phi_dL2)

    # --- プロット ---
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(L_log, dphi_dL, color="blue")
    ax[0].set_xscale("log")
    ax[0].set_ylabel(r"$\partial \Phi_t / \partial L$")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(L_log, d2phi_dL2, color="green")
    if L_zero is not None:
        ax[1].axvline(L_zero, color="red", linestyle="--", label=f"L₍c⊕₎ ≈ {L_zero:.2e} m")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("L [m]")
    ax[1].set_ylabel(r"$\partial^2 \Phi_t / \partial L^2$")
    ax[1].grid(True, alpha=0.3)
    if L_zero is not None:
        ax[1].legend()

    fig.suptitle("Time-field curvature scan (log–log interpolation)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{args.out}_curvature_logscan.png", dpi=200)

    # --- 結果出力 ---
    summary = {
        "L_zero_curvature_estimate_m": L_zero,
        "L_min": float(L_log.min()),
        "L_max": float(L_log.max()),
    }
    pd.DataFrame([summary]).to_csv(f"{args.out}_logscan_summary.csv", index=False)

    if L_zero is not None:
        print(f"Zero-curvature point detected: L₍c⊕₎ ≈ {L_zero:.3e} m")
    else:
        print("No sign change detected in curvature (∂²Φₜ/∂L²).")


if __name__ == "__main__":
    main()
