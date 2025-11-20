#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase40_phi_t_mass_map.py

Phase 40:
  Phase 37 (TFGR 二階層) と Phase 38 (質量スケーリング Lc_Q ∝ 1/m_eff)
  の結果を統合し、質量依存時間場ポテンシャル Φ_t(L, m) の
  2D マップを可視化する。

前提：
  - phase36_multiscale_dataset.csv      : 元データ（L_macro_m, L_int_m など）
  - output_phase37/waic_summary_phase37.txt : マクロ側 TFGR パラメータ
  - output_phase38_mass/waic_mass_scaling.txt : 量子側 TFGR+質量スケーリング

出力：
  - phi_t_mass_map.png       : (log10 L, m_eff) 空間の Φ_t ヒートマップ
  - phi_t_mass_slices.png    : 代表的質量での Δt(L) スライス
  - phi_t_mass_summary.txt   : 使用したパラメータ一覧
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------  Phase 37 パラメータのパーサ  --------
def parse_phase37_params(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def grab(name):
        m = re.search(rf"{name}\s*=\s*([\-0-9.eE\+]+)", text)
        return float(m.group(1)) if m else np.nan

    dt0_M = grab("dt0_M")
    Lc_M  = grab("Lc_M")
    p_M   = grab("p_M")
    q_M   = grab("q_M")
    dt0_Q = grab("dt0_Q")
    Lc_Q  = grab("Lc_Q")
    p_Q   = grab("p_Q")
    q_Q   = grab("q_Q")

    return {
        "dt0_M": dt0_M,
        "Lc_M": Lc_M,
        "p_M": p_M,
        "q_M": q_M,
        "dt0_Q_phase37": dt0_Q,
        "Lc_Q_phase37": Lc_Q,
        "p_Q_phase37": p_Q,
        "q_Q_phase37": q_Q,
    }


# --------  Phase 38 パラメータのパーサ  --------
def parse_phase38_mass_params(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # LcQ0, dt0_Q, p_Q, q_Q
    def grab(name_pattern):
        m = re.search(name_pattern + r".*?=\s*([\-0-9.eE\+]+)", text)
        return float(m.group(1)) if m else np.nan

    LcQ0 = grab(r"LcQ0")
    dt0Q = grab(r"dt0_Q")
    pQ   = grab(r"p_Q")
    qQ   = grab(r"q_Q")

    # Effective masses も取っておく（参考）
    masses = {}
    in_block = False
    for line in text.splitlines():
        if "Effective masses per pair" in line:
            in_block = True
            continue
        if in_block and line.strip():
            m = re.search(r"([\w\-\+_]+):.*?([\d\.]+)\s*u", line)
            if m:
                pair = m.group(1)
                val = float(m.group(2))
                masses[pair] = val

    return {
        "LcQ0": LcQ0,
        "dt0_Q": dt0Q,
        "p_Q": pQ,
        "q_Q": qQ,
        "masses": masses,
    }


# --------  TFGR Δt(L, m) の定義  --------
def delta_t_tfgr(L, L_int_ref, m_eff,
                 dt0_M, Lc_M, p_M, q_M,
                 dt0_Q, LcQ0, p_Q, q_Q,
                 m0=100.0):
    """
    L [m] : スケール（ここではマクロスケールとして扱う）
    L_int_ref [m] : 参照内部スケール（例えば 5 mm）
    m_eff [u] : 有効質量
    """
    # マクロ側
    term_M = dt0_M * (1.0 + (L / Lc_M) ** p_M) ** q_M

    # 質量スケーリング付き量子側 Lc_Q(m)
    Lc_Q_m = LcQ0 * (m0 / m_eff)
    term_Q = dt0_Q * (1.0 + (L_int_ref / Lc_Q_m) ** p_Q) ** q_Q

    return term_M + term_Q, term_M, term_Q, Lc_Q_m


def main():
    ap = argparse.ArgumentParser(
        description="Phase 40: mass-dependent time-field potential map."
    )
    ap.add_argument("--csv", required=True,
                    help="phase36_multiscale_dataset.csv")
    ap.add_argument("--phase37", required=True,
                    help="output_phase37/waic_summary_phase37.txt")
    ap.add_argument("--phase38", required=True,
                    help="output_phase38_mass/waic_mass_scaling.txt")
    ap.add_argument("--out", default="output_phase40_phi_t",
                    help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- データ読み込み（L_int の代表値を取得） ---
    df = pd.read_csv(args.csv)
    if "L_int_m" not in df.columns:
        raise ValueError("CSV に L_int_m 列が必要です。")
    L_int_ref = df["L_int_m"].median()  # 代表値（≈ 5 mm）

    # --- パラメータ読み込み ---
    p37 = parse_phase37_params(args.phase37)
    p38 = parse_phase38_mass_params(args.phase38)

    dt0_M = p37["dt0_M"]
    Lc_M  = p37["Lc_M"]
    p_M   = p37["p_M"]
    q_M   = p37["q_M"]

    # 量子側は Phase 38 の質量スケーリング版を使う
    dt0_Q = p38["dt0_Q"]
    LcQ0  = p38["LcQ0"]
    p_Q   = p38["p_Q"]
    q_Q   = p38["q_Q"]

    # --- マップ用のグリッドを定義 ---
    # L: 10^-3 ～ 10^8 m（量子～惑星スケール）
    logL_min, logL_max = -3, 8
    nL = 200
    L_vals = np.logspace(logL_min, logL_max, nL)

    # m_eff: 80 ～ 200 u（Sr ～ Yb, In あたり）
    m_min, m_max = 80.0, 200.0
    nm = 120
    m_vals = np.linspace(m_min, m_max, nm)

    # 2D グリッド計算
    Phi = np.zeros((nm, nL))
    Phi_M = np.zeros_like(Phi)
    Phi_Q = np.zeros_like(Phi)
    LcQ_map = np.zeros_like(Phi)

    for i, m_eff in enumerate(m_vals):
        for j, L in enumerate(L_vals):
            dt_tot, dt_M, dt_Q, LcQ_m = delta_t_tfgr(
                L, L_int_ref, m_eff,
                dt0_M, Lc_M, p_M, q_M,
                dt0_Q, LcQ0, p_Q, q_Q,
            )
            Phi[i, j] = dt_tot
            Phi_M[i, j] = dt_M
            Phi_Q[i, j] = dt_Q
            LcQ_map[i, j] = LcQ_m

    # 全体を代表スケールで正規化して「Φ_t」とする
    Phi_norm = Phi / np.max(np.abs(Phi))

    # --- ヒートマップ描画 ---
    X, Y = np.meshgrid(np.log10(L_vals), m_vals)

    plt.figure(figsize=(7, 5))
    im = plt.pcolormesh(
        X, Y, Phi_norm,
        shading="auto"
    )
    plt.colorbar(im, label="normalized Φ_t (Δt / max|Δt|)")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Mass-dependent Time-Field Potential Φ_t(L, m)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "phi_t_mass_map.png"), dpi=300)
    plt.close()

    # --- 代表的質量での Δt(L) スライス ---
    masses_for_slice = [88.0, 115.0, 171.0]  # Sr, In, Yb 想定
    plt.figure(figsize=(7, 5))
    for m_eff in masses_for_slice:
        dt_tot, dt_M, dt_Q, _ = delta_t_tfgr(
            L_vals, L_int_ref, m_eff,
            dt0_M, Lc_M, p_M, q_M,
            dt0_Q, LcQ0, p_Q, q_Q,
        )
        plt.plot(L_vals, dt_tot, label=f"m={m_eff:.0f} u")
    plt.xscale("log")
    plt.xlabel("L [m]")
    plt.ylabel("Δt_TFGR(L, m) [s]")
    plt.title("TFGR Δt(L) for representative masses")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "phi_t_mass_slices.png"), dpi=300)
    plt.close()

    # --- まとめテキスト ---
    summary_path = os.path.join(args.out, "phi_t_mass_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 40: mass-dependent time-field potential Φ_t(L, m)\n\n")
        f.write("Macro-scale TFGR parameters (from Phase 37):\n")
        f.write(f"  dt0_M = {dt0_M:.3e} [s]\n")
        f.write(f"  Lc_M  = {Lc_M:.3e} [m]\n")
        f.write(f"  p_M   = {p_M:.3f}\n")
        f.write(f"  q_M   = {q_M:.3f}\n\n")
        f.write("Quantum-scale TFGR parameters with mass scaling (from Phase 38):\n")
        f.write(f"  dt0_Q = {dt0_Q:.3e} [s]\n")
        f.write(f"  LcQ0  = {LcQ0:.3e} [m]   (for m0 = 100 u)\n")
        f.write(f"  p_Q   = {p_Q:.3f}\n")
        f.write(f"  q_Q   = {q_Q:.3f}\n\n")
        f.write(f"Representative internal scale L_int_ref = {L_int_ref:.3e} [m]\n")

    print(f"✅ phi_t_mass_map.png, phi_t_mass_slices.png, phi_t_mass_summary.txt を {args.out} に出力しました。")


if __name__ == "__main__":
    main()
