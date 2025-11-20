#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase45A_tfgr_schwarzschild.py

Phase 45A:
  Schwarzschild 近似の地球重力場において、
  通常の GR 重力赤方偏移 y_GR(r) と
  TFGR による追加周波数シフト y_TFGR(L,m) を比較する。

前提:
  - Phase 40 の結果ファイル:
        output_phase40_phi_t/phi_t_mass_summary.txt
    が存在していること。
  - L_macro スケールに対する TFGR パラメータを
    「系全体の代表スケール L」とみなして使う。

出力:
  - output_phase45A_tfgr_schwarzschild/
        freq_shift_vs_r.png   : y_GR, y_GR+TFGR の r 依存プロット
        tfgr_vs_gr_summary.txt : 代表高度での比較サマリ

使い方の例:
  python phase45A_tfgr_schwarzschild.py
  python phase45A_tfgr_schwarzschild.py --m_eff 171 --Tref 86400
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 物理定数
C = 2.99792458e8      # [m/s]
G = 6.67430e-11       # [m^3 kg^-1 s^-2]
M_EARTH = 5.97219e24  # [kg]
R_EARTH = 6.371e6     # [m]


# ---------- Phase 40 パラメータ読込 ----------

def load_phase40_params(summary_path):
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
    """
    Phase 40 と同じ Δt_TFGR(L,m) モデル。
    L: マクロスケール [m]
    m_eff: 有効質量 [u]
    """
    # Macro part
    term_M = dt0_M * (1.0 + (L / Lc_M) ** p_M) ** q_M
    # Quantum part with mass scaling
    Lc_Q_m = LcQ0 * (m0 / m_eff)
    term_Q = dt0_Q * (1.0 + (L_int_ref / Lc_Q_m) ** p_Q) ** q_Q
    return term_M + term_Q


# ---------- GR の重力赤方偏移 ----------

def freq_shift_GR(r, r_ref=R_EARTH, M=M_EARTH):
    """
    静止している時計の GR 重力赤方偏移（弱い場近似ではない「正確な」Schwarzschild形）。
    周波数比:
        nu(r) / nu(r_ref) = sqrt( (1 - 2GM/(r c^2)) / (1 - 2GM/(r_ref c^2)) )
    から、
        y_GR = nu(r)/nu(r_ref) - 1
    を返す。
    """
    rs = 2.0 * G * M / C**2
    factor = np.sqrt((1.0 - rs / r) / (1.0 - rs / r_ref))
    return factor - 1.0   # dimensionless


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compare GR redshift and TFGR time correction in Earth Schwarzschild field."
    )
    parser.add_argument(
        "--summary",
        default=os.path.join("output_phase40_phi_t", "phi_t_mass_summary.txt"),
        help="Path to phi_t_mass_summary.txt (Phase 40 result).",
    )
    parser.add_argument(
        "--out",
        default="output_phase45A_tfgr_schwarzschild",
        help="Output directory.",
    )
    parser.add_argument(
        "--m_eff",
        type=float,
        default=88.0,
        help="Effective atomic mass m_eff [u], e.g. 88 (Sr), 171 (Yb).",
    )
    parser.add_argument(
        "--Tref",
        type=float,
        default=86400.0,
        help="Reference integration time T_ref [s] for converting Δt_TFGR to fractional shift (default: 1 day).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- パラメータ読み込み ----
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

    # ---- r グリッド定義 ----
    # 地表～月軌道までざっくり
    r_vals = np.linspace(R_EARTH, 4.0e8, 400)  # [m]

    # GR 重力赤方偏移
    y_GR = freq_shift_GR(r_vals, r_ref=R_EARTH, M=M_EARTH)

    # TFGR 部分:
    #   「系の代表スケール」を L ≈ r とみなして Δt_TFGR(L,m) を評価
    L_macro = r_vals
    dt_tfgr = delta_t_tfgr(
        L_macro, L_int, args.m_eff,
        dt0_M, Lc_M, p_M, q_M,
        dt0_Q, LcQ0, p_Q, q_Q
    )  # [s]

    # 周波数シフトとしての TFGR 寄与
    y_TFGR = dt_tfgr / args.Tref   # dimensionless

    # 合成
    y_total = y_GR + y_TFGR

    # ---- 図を作成 ----
    plt.figure(figsize=(7, 5))
    # r を地表からの高度 h = r - R_E で表示
    h_vals = r_vals - R_EARTH

    plt.plot(h_vals/1e3, y_GR, label="GR only")
    plt.plot(h_vals/1e3, y_total, label="GR + TFGR")
    plt.xlabel("Altitude h above Earth surface [km]")
    plt.ylabel("Fractional frequency shift y")
    plt.title(f"GR vs GR+TFGR (m_eff={args.m_eff:.1f} u, T_ref={args.Tref:.0f} s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(args.out, "freq_shift_vs_r.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()

    # ---- 代表高度での数値サマリ ----
    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    # 代表点: 地表, GPS, 静止軌道, 月軌道
    r_surface = R_EARTH
    r_gps     = R_EARTH + 2.02e7     # ~ 20,200 km
    r_geo     = 4.2164e7             # geostationary orbit radius
    r_moon    = 3.844e8              # mean Earth-Moon distance

    points = [
        ("surface", r_surface),
        ("GPS",     r_gps),
        ("GEO",     r_geo),
        ("Moon",    r_moon),
    ]

    summary_path = os.path.join(args.out, "tfgr_vs_gr_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 45A: GR vs GR+TFGR frequency shift in Earth Schwarzschild field\n\n")
        f.write(f"m_eff = {args.m_eff:.1f} u\n")
        f.write(f"T_ref = {args.Tref:.3e} s\n\n")
        f.write("Columns: y_GR, y_TFGR, y_total, ratio = y_TFGR / y_GR\n\n")

        for name, r0 in points:
            i = nearest_idx(r_vals, r0)
            y_gr   = y_GR[i]
            y_tfgr = y_TFGR[i]
            y_tot  = y_total[i]
            ratio  = y_tfgr / y_gr if y_gr != 0 else np.nan

            f.write(f"[{name}]: r = {r_vals[i]:.3e} m, h = {r_vals[i]-R_EARTH:.3e} m\n")
            f.write(f"  y_GR    = {y_gr:.6e}\n")
            f.write(f"  y_TFGR  = {y_tfgr:.6e}\n")
            f.write(f"  y_total = {y_tot:.6e}\n")
            f.write(f"  y_TFGR / y_GR ≈ {ratio:.3e}\n\n")

    print("✅ 出力完了:")
    print("  図:", out_fig)
    print("  サマリ:", summary_path)


if __name__ == "__main__":
    main()
