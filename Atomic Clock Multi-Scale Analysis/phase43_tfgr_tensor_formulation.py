#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase43_tfgr_tensor_formulation.py

Phase 43:
  TFGR 時間場を GR に組み込むための
  有効「時間曲率」R_t(L,m) とエネルギー密度 rho_t(L,m) を計算・可視化。

入力:
  - output_phase40_phi_t/phi_t_mass_summary.txt

出力:
  - Rt_map.png                : R_t(L,m) のヒートマップ
  - rho_t_map.png             : 有効エネルギー密度 rho_t(L,m) のヒートマップ
  - tfgr_tensor_summary.txt   : 代表スケールでの R_t, rho_t の数値
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 物理定数
C = 2.99792458e8      # [m/s]
G = 6.67430e-11       # [m^3 kg^-1 s^-2]
PI = np.pi

def load_params(summary_path):
    txt = open(summary_path, "r", encoding="utf-8").read()
    def grab(name):
        m = re.search(rf"{name}\s*=\s*([\-0-9.eE\+]+)", txt)
        return float(m.group(1)) if m else np.nan

    p = {}
    p["dt0_M"] = grab("dt0_M")
    p["Lc_M"]  = grab("Lc_M")
    p["p_M"]   = grab("p_M")
    p["q_M"]   = grab("q_M")
    p["dt0_Q"] = grab("dt0_Q")
    p["LcQ0"]  = grab("LcQ0")
    p["p_Q"]   = grab("p_Q")
    p["q_Q"]   = grab("q_Q")
    p["L_int_ref"] = grab("L_int_ref")
    return p

def delta_t_tfgr(L, L_int_ref, m_eff,
                 dt0_M, Lc_M, p_M, q_M,
                 dt0_Q, LcQ0, p_Q, q_Q,
                 m0=100.0):
    # Macro part
    term_M = dt0_M * (1.0 + (L / Lc_M)**p_M)**q_M
    # Quantum, mass-scaled
    Lc_Q_m = LcQ0 * (m0 / m_eff)
    term_Q = dt0_Q * (1.0 + (L_int_ref / Lc_Q_m)**p_Q)**q_Q
    return term_M + term_Q

def main():
    summary_path = os.path.join("output_phase40_phi_t", "phi_t_mass_summary.txt")
    out_dir = "output_phase43_tfgr_tensor"
    os.makedirs(out_dir, exist_ok=True)

    params = load_params(summary_path)
    dt0_M   = params["dt0_M"]
    Lc_M    = params["Lc_M"]
    p_M     = params["p_M"]
    q_M     = params["q_M"]
    dt0_Q   = params["dt0_Q"]
    LcQ0    = params["LcQ0"]
    p_Q     = params["p_Q"]
    q_Q     = params["q_Q"]
    L_int   = params["L_int_ref"]

    # ----- グリッド定義 -----
    L_vals = np.logspace(-3, 8, 250)    # 10^-3 ～ 10^8 m
    m_vals = np.linspace(80.0, 200.0, 140)  # 80 ～ 200 u

    logL = np.log10(L_vals)
    dlogL = np.gradient(logL)
    dm = np.gradient(m_vals)

    Phi = np.zeros((len(m_vals), len(L_vals)))
    for i, m_eff in enumerate(m_vals):
        for j, L in enumerate(L_vals):
            Phi[i, j] = delta_t_tfgr(
                L, L_int, m_eff,
                dt0_M, Lc_M, p_M, q_M,
                dt0_Q, LcQ0, p_Q, q_Q
            )

    # Δt をそのまま Φ_t の代理として扱う（10^-15 s オーダー）
    Phi_norm = Phi / np.max(np.abs(Phi))

    # ----- 時間曲率 R_t ≈ ∇^2 Φ_t を計算 -----
    d2Phi_dL2 = np.gradient(
        np.gradient(Phi_norm, axis=1), axis=1
    ) / (dlogL**2)

    d2Phi_dm2 = np.gradient(
        np.gradient(Phi_norm, axis=0), axis=0
    ) / (dm[:, None]**2)

    # ラプラシアン（次元レス）
    Rt_dimless = d2Phi_dL2 + d2Phi_dm2

    # 代表長 L0 を Lc_M として、次元付き R_t を定義
    L0 = Lc_M
    Rt = Rt_dimless / (L0**2)    # [1/m^2] を想定

    # 有効エネルギー密度 rho_t (かなりモデル依存な試験定義)
    rho_t = (C**4 / (8.0 * PI * G)) * Rt   # [J/m^3]

    X, Y = np.meshgrid(logL, m_vals)

    # ----- R_t ヒートマップ -----
    plt.figure(figsize=(7,5))
    im = plt.pcolormesh(X, Y, Rt, cmap="coolwarm", shading="auto")
    plt.colorbar(im, label="R_t(L,m) [1/m^2] (up to scaling)")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Effective time-curvature scalar R_t(L, m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Rt_map.png"), dpi=300)
    plt.close()

    # ----- rho_t ヒートマップ（オーダー把握用に log10 表示） -----
    # 正負をまたぐので、絶対値をとって log10 に
    rho_abs = np.abs(rho_t) + 1e-50
    log_rho = np.log10(rho_abs)

    plt.figure(figsize=(7,5))
    im2 = plt.pcolormesh(X, Y, log_rho, cmap="viridis", shading="auto")
    plt.colorbar(im2, label="log10 |rho_t| [J/m^3]")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Effective time-field energy density |rho_t(L, m)|")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rho_t_map.png"), dpi=300)
    plt.close()

    # ----- 代表点での値を出力 -----
    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    L_quantum = 1e-3   # 量子スケール
    L_gps     = 2e7    # GPS 軌道高度 ~ 2e7 m
    L_lunar   = 4e8    # 地球-月距離オーダー

    points = [
        ("quantum", L_quantum,  88.0),
        ("gps",     L_gps,      88.0),
        ("lunar",   L_lunar,    88.0),
        ("lunar_Yb",L_lunar,   171.0),
    ]

    summary_path = os.path.join(out_dir, "tfgr_tensor_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 43: TFGR effective time-curvature and energy density\n\n")
        f.write(f"L0 (reference length) = {L0:.3e} m\n\n")
        for name, L0p, mp in points:
            i = nearest_idx(m_vals, mp)
            j = nearest_idx(L_vals, L0p)
            rt_val  = Rt[i, j]
            rho_val = rho_t[i, j]
            f.write(f"[{name}]: L={L_vals[j]:.3e} m, m_eff={m_vals[i]:.1f} u\n")
            f.write(f"  R_t  ≈ {rt_val:.3e} [1/m^2]\n")
            f.write(f"  rho_t≈ {rho_val:.3e} [J/m^3]\n\n")

    print("✅ 出力完了:")
    print("   ", os.path.join(out_dir, "Rt_map.png"))
    print("   ", os.path.join(out_dir, "rho_t_map.png"))
    print("   ", summary_path)

if __name__ == "__main__":
    main()
