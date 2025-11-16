#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 177 修正版
TFGR feedback curvature (Phi_feedback) から
「実効ダークエネルギー密度」を構成し、
Pantheon SN データに対して χ² 最小化を行う。

改良点：
- Phi_feedback の列名を自動判定（ゆるいマッチング）
- BOM / 不可視文字の除去
- 欠損値クリーニング
"""

import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# =========================================================
# ★ Phi_feedback の列名を自動検出（柔軟マッチング）
# =========================================================
def find_phi_feedback_column(df):
    cleaned = [c.strip().replace("\ufeff", "").lower() for c in df.columns]
    candidates = []

    for orig, c in zip(df.columns, cleaned):
        if "phi" in c and "feed" in c:
            candidates.append(orig)

    if len(candidates) == 0:
        raise RuntimeError(
            f"Phi_feedback 列が見つかりません。利用可能な列: {df.columns.tolist()}"
        )

    print(f"[INFO] Phi_feedback 列を自動検出: {candidates[0]}")
    return candidates[0]


# =========================================================
# TFGR feedback CSV の読み込み
# =========================================================
def load_tfgr_feedback(csv_path):
    df = pd.read_csv(csv_path)

    # 列名の不可視文字・前後スペース除去
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # z 列の特定
    z_col = None
    for c in df.columns:
        if c.strip().lower() == "z":
            z_col = c
            break
    if z_col is None:
        raise RuntimeError("z列が見つかりません。")

    # Phi_feedback の列探索
    phi_col = find_phi_feedback_column(df)

    z = df[z_col].values.astype(float)
    phi = df[phi_col].values.astype(float)

    # 欠損除去
    mask = np.isfinite(z) & np.isfinite(phi)
    z = z[mask]
    phi = phi[mask]

    # 昇順に並べ替え
    idx = np.argsort(z)
    return z[idx], phi[idx]


# =========================================================
# Pantheon SN の読み込み
# =========================================================
def load_sn(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    z = df["z"].values
    mu = df["mu"].values
    mu_err = df["mu_err"].values

    mask = np.isfinite(z) & np.isfinite(mu) & np.isfinite(mu_err)
    return z[mask], mu[mask], mu_err[mask]


# =========================================================
# H(z) モデル（TFGR 追加 DE を含む）
# =========================================================
def hubble_model(z, H0, Om0, Or0, OmDE_z):
    return H0 * np.sqrt(Or0 * (1 + z) ** 4 + Om0 * (1 + z) ** 3 + OmDE_z)


# =========================================================
# d_L(z) の数値積分
# =========================================================
def luminosity_distance(z_arr, H0, Om0, Or0, z_grid, OmDE_grid):
    # Ω_DE(z) を SN の z_arr へ補間
    OmDE_interp = interp1d(z_grid, OmDE_grid, kind="linear", fill_value="extrapolate")
    OmDE_sn = OmDE_interp(z_arr)

    # H(z) の補間
    H_sn = H0 * np.sqrt(Or0 * (1 + z_arr)**4 + Om0 * (1 + z_arr)**3 + OmDE_sn)

    # 数値積分で dL(z) を求める
    d_integral = []
    for z in z_arr:
        z_int = np.linspace(0, z, 400)
        H_int = H0 * np.sqrt(
            Or0*(1+z_int)**4 + Om0*(1+z_int)**3 + OmDE_interp(z_int)
        )
        val = np.trapz(1.0 / H_int, z_int)
        d_integral.append(val)

    c = 299792.458
    return (1 + z_arr) * c * np.array(d_integral)



def mu_from_dL(dL):
    return 5 * np.log10(dL) + 25


# =========================================================
# TFGR feedback → Ω_DE(z)
# =========================================================
def construct_OmDE_from_feedback(z, phi_fb, eps):
    """ Ω_DE(z) = exp( eps * Φ_fb(z) ) """
    return np.exp(eps * (phi_fb - phi_fb.min()))


# =========================================================
# χ²
# =========================================================
def chi2_calc(mu_obs, mu_err, mu_model):
    return np.sum(((mu_obs - mu_model) / mu_err) ** 2)


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfgr_csv", required=True)
    parser.add_argument("--sn_csv", required=True)
    parser.add_argument("--H0", type=float, default=70.0)
    parser.add_argument("--Omega_m0", type=float, default=0.3)
    parser.add_argument("--Omega_r0", type=float, default=1e-4)
    parser.add_argument("--eps_min", type=float, default=-5.0)
    parser.add_argument("--eps_max", type=float, default=5.0)
    parser.add_argument("--eps_steps", type=int, default=401)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    print("=== Phase 177: TFGR feedback-based DE + SN fit ===")

    # load TFGR feedback
    z_fb, phi_fb = load_tfgr_feedback(args.tfgr_csv)

    # load SN
    z_sn, mu_sn, mu_err = load_sn(args.sn_csv)

    # 統一 z グリッド
    z_grid = np.linspace(0, max(z_sn.max(), z_fb.max()), 600)

    # feedback 補間
    phi_interp = interp1d(z_fb, phi_fb, kind="linear", fill_value="extrapolate")

    # chi2 scan
    eps_vals = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    chi2_list = []

    H0 = args.H0
    Om0 = args.Omega_m0
    Or0 = args.Omega_r0

    for eps in eps_vals:
        OmDE_z = construct_OmDE_from_feedback(z_grid, phi_interp(z_grid), eps)

        dL = luminosity_distance(z_sn, H0, Om0, Or0, z_grid, OmDE_z)
        mu_model = mu_from_dL(dL)

        chi2 = chi2_calc(mu_sn, mu_err, mu_model)
        chi2_list.append(chi2)

    chi2_arr = np.array(chi2_list)
    idx_best = np.argmin(chi2_arr)
    eps_best = eps_vals[idx_best]
    chi2_min = chi2_arr[idx_best]
    dof = len(z_sn) - 1
    chi2_red = chi2_min / dof

    print("------ BEST FIT ------")
    print(f"eps_best = {eps_best:.6f}")
    print(f"chi2_min = {chi2_min:.3f}")
    print(f"dof      = {dof}")
    print(f"chi2_red = {chi2_red:.6f}")
    print("----------------------")

    # chi2 vs eps plot
    plt.figure(figsize=(8, 5))
    plt.plot(eps_vals, chi2_arr)
    plt.axvline(eps_best, color="r", linestyle="--")
    plt.xlabel("eps")
    plt.ylabel("chi2")
    plt.title("Phase 177: chi2 vs eps")
    plt.grid()
    plt.savefig(f"{args.out_prefix}_chi2_vs_eps.png", dpi=150)

    # 保存
    pd.DataFrame({
        "eps_best": [eps_best],
        "chi2_min": [chi2_min],
        "dof": [dof],
        "chi2_red": [chi2_red]
    }).to_csv(f"{args.out_prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
