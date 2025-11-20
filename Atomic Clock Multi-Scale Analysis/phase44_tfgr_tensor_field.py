#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase44_tfgr_tensor_field.py

Phase 44:
  時間場曲率 R_t(L,m) から有効時間場テンソル
  T^{(t)}_{μν} = diag( rho_t c^2, p_t, p_t, p_t )
  を構成し、マップと数値サマリを出力するスクリプト。

前提:
  - Phase 40 の結果ファイル:
        output_phase40_phi_t/phi_t_mass_summary.txt
    が存在していること。

出力:
  - output_phase44_tfgr_tensor/
        T00_map.png           : T^{(t)}_{00}(L,m) の log10 ヒートマップ
        p_t_map.png           : p_t(L,m) の log10 ヒートマップ
        tfgr_tensor_field.npz : L, m, rho_t, p_t, T00, Tii を保存した NumPy アーカイブ
        tfgr_tensor_field_summary.txt : 代表スケールでの値などのサマリ

使い方の例:
  python phase44_tfgr_tensor_field.py
  python phase44_tfgr_tensor_field.py --w_t -1.0 --out output_phase44_tfgr_tensor
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 物理定数
C = 2.99792458e8      # [m/s]
G = 6.67430e-11       # [m^3 kg^-1 s^-2]
PI = np.pi


# ---------- ユーティリティ ----------

def load_phase40_params(summary_path):
    """Phase 40 の phi_t_mass_summary.txt からパラメータを取得"""
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()

    def grab(name):
        m = re.search(rf"{name}\s*=\s*([\-0-9.eE\+]+)", text)
        return float(m.group(1)) if m else np.nan

    p = {}
    p["dt0_M"]    = grab("dt0_M")
    p["Lc_M"]     = grab("Lc_M")
    p["p_M"]      = grab("p_M")
    p["q_M"]      = grab("q_M")
    p["dt0_Q"]    = grab("dt0_Q")
    p["LcQ0"]     = grab("LcQ0")
    p["p_Q"]      = grab("p_Q")
    p["q_Q"]      = grab("q_Q")
    p["L_int_ref"] = grab("L_int_ref")
    return p


def delta_t_tfgr(L, L_int_ref, m_eff,
                 dt0_M, Lc_M, p_M, q_M,
                 dt0_Q, LcQ0, p_Q, q_Q,
                 m0=100.0):
    """Phase 40 と同じ TFGR Δt(L,m) 定義"""
    # Macro part
    term_M = dt0_M * (1.0 + (L / Lc_M) ** p_M) ** q_M
    # Quantum part with mass scaling
    Lc_Q_m = LcQ0 * (m0 / m_eff)
    term_Q = dt0_Q * (1.0 + (L_int_ref / Lc_Q_m) ** p_Q) ** q_Q
    return term_M + term_Q


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(
        description="Construct TFGR time-field tensor T^{(t)}_{μν} on (L, m) grid."
    )
    parser.add_argument(
        "--summary",
        default=os.path.join("output_phase40_phi_t", "phi_t_mass_summary.txt"),
        help="Path to phi_t_mass_summary.txt from Phase 40",
    )
    parser.add_argument(
        "--out",
        default="output_phase44_tfgr_tensor",
        help="Output directory",
    )
    parser.add_argument(
        "--w_t",
        type=float,
        default=-1.0,
        help="Equation-of-state parameter w_t in p_t = w_t * rho_t (default: -1.0)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- パラメータ読み込み ---
    params = load_phase40_params(args.summary)
    dt0_M   = params["dt0_M"]
    Lc_M    = params["Lc_M"]
    p_M     = params["p_M"]
    q_M     = params["q_M"]
    dt0_Q   = params["dt0_Q"]
    LcQ0    = params["LcQ0"]
    p_Q     = params["p_Q"]
    q_Q     = params["q_Q"]
    L_int   = params["L_int_ref"]

    # --- グリッド定義 ---
    L_vals = np.logspace(-3, 8, 250)       # 10^-3 ～ 10^8 m
    m_vals = np.linspace(80.0, 200.0, 140) # 80 ～ 200 u

    logL = np.log10(L_vals)
    dlogL = np.gradient(logL)
    dm = np.gradient(m_vals)

    # --- Φ_t(L,m) = Δt_TFGR を計算 ---
    Phi = np.zeros((len(m_vals), len(L_vals)))
    for i, m_eff in enumerate(m_vals):
        for j, L in enumerate(L_vals):
            Phi[i, j] = delta_t_tfgr(
                L, L_int, m_eff,
                dt0_M, Lc_M, p_M, q_M,
                dt0_Q, LcQ0, p_Q, q_Q
            )

    # 規格化（次元レス）
    Phi_norm = Phi / np.max(np.abs(Phi))

    # --- 時間曲率 R_t ≈ ∇^2 Φ_t を計算（Phase 43 と同様の手順） ---
    d2Phi_dL2 = np.gradient(
        np.gradient(Phi_norm, axis=1), axis=1
    ) / (dlogL ** 2)

    d2Phi_dm2 = np.gradient(
        np.gradient(Phi_norm, axis=0), axis=0
    ) / (dm[:, None] ** 2)

    Rt_dimless = d2Phi_dL2 + d2Phi_dm2  # 無次元ラプラシアン

    # 代表長さ L0 を Lc_M とする
    L0 = Lc_M
    Rt = Rt_dimless / (L0 ** 2)  # [1/m^2]（スケーリングを仮定）

    # --- 有効エネルギー密度 rho_t とテンソル成分 ---
    rho_t = (C ** 4 / (8.0 * PI * G)) * Rt      # [J/m^3]
    p_t   = args.w_t * rho_t                    # [J/m^3]

    # テンソル成分（座標系 diag(-,+,+,+) を想定）
    T00 = rho_t * C ** 2        # [J/m^3] * c^2 → [J/m^3 * (m^2/s^2)] = [N/m^2] ~ [Pa]
    Tii = p_t                   # 空間対角成分（x,y,z で同一）

    X, Y = np.meshgrid(logL, m_vals)

    # --- 図1: T00 マップ（log10 |T00|） ---
    T00_abs = np.abs(T00) + 1e-50
    log_T00 = np.log10(T00_abs)

    plt.figure(figsize=(7, 5))
    im1 = plt.pcolormesh(X, Y, log_T00, cmap="viridis", shading="auto")
    plt.colorbar(im1, label="log10 |T^{(t)}_{00}|  [SI units, up to scaling]")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("TFGR time-field tensor component T^{(t)}_{00}(L,m)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "T00_map.png"), dpi=300)
    plt.close()

    # --- 図2: p_t マップ（log10 |p_t|） ---
    p_abs = np.abs(p_t) + 1e-50
    log_p = np.log10(p_abs)

    plt.figure(figsize=(7, 5))
    im2 = plt.pcolormesh(X, Y, log_p, cmap="plasma", shading="auto")
    plt.colorbar(im2, label="log10 |p_t|  [J/m^3]")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("TFGR effective time-pressure p_t(L,m)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "p_t_map.png"), dpi=300)
    plt.close()

    # --- NumPy アーカイブとして保存 ---
    npz_path = os.path.join(args.out, "tfgr_tensor_field.npz")
    np.savez(
        npz_path,
        L=L_vals,
        m=m_vals,
        Phi=Phi,
        Rt=Rt,
        rho_t=rho_t,
        p_t=p_t,
        T00=T00,
        Tii=Tii,
        w_t=args.w_t,
        L0=L0,
    )

    # --- 代表スケールでの値をテキストで出力 ---
    points = [
        ("quantum", 1.0e-3, 88.0),
        ("gps",     2.0e7, 88.0),
        ("lunar_Sr",1.0e8, 88.0),
        ("lunar_Yb",1.0e8,171.0),
    ]

    summary_path = os.path.join(args.out, "tfgr_tensor_field_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 44: TFGR time-field tensor T^{(t)}_{μν}\n\n")
        f.write(f"Equation-of-state parameter w_t = {args.w_t:.3f}\n")
        f.write(f"Reference length L0 = Lc_M = {L0:.3e} m\n\n")
        for name, L0p, mp in points:
            i = nearest_idx(m_vals, mp)
            j = nearest_idx(L_vals, L0p)
            f.write(f"[{name}]: L = {L_vals[j]:.3e} m, m_eff = {m_vals[i]:.1f} u\n")
            f.write(f"  R_t   ≈ {Rt[i, j]:.3e} [1/m^2]\n")
            f.write(f"  rho_t ≈ {rho_t[i, j]:.3e} [J/m^3]\n")
            f.write(f"  T00   ≈ {T00[i, j]:.3e} [SI units]\n")
            f.write(f"  p_t   ≈ {p_t[i, j]:.3e} [J/m^3]\n\n")

    print("✅ 出力完了:")
    print("  ", os.path.join(args.out, "T00_map.png"))
    print("  ", os.path.join(args.out, "p_t_map.png"))
    print("  ", npz_path)
    print("  ", summary_path)


if __name__ == "__main__":
    main()
