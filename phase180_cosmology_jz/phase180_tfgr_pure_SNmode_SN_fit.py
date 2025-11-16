#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 180: Pure TFGR (matter + SN-mode time field) SN fit
---------------------------------------------------------

H^2(z) = H0^2 [ Omega_r0 (1+z)^4
               + Omega_m0 (1+z)^3
               + Omega_SN0 * f_SN(z) ]

ここで
  * f_SN(z) = rho_SN(z) / rho_SN(0)
  * 正規化条件として z=0 で H(z)=H0 となるように
        Omega_SN0 = 1 - Omega_r0 - Omega_m0
  * 従って自由パラメータは Omega_m0 のみ（C はSNのゼロ点補正）

使用ファイル:
  - SN: Pantheon 距離 (pantheon_SN.csv など)
  - SN モード時間場エネルギー: phase178_tfgr_SN_mode_rho_SN_profile.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C_LIGHT_KM_S = 299792.458  # speed of light [km/s]


def load_sn_csv(sn_csv, z_min=0.0, z_max=2.0):
    df = pd.read_csv(sn_csv)

    # z 列の自動検出
    if "z" in df.columns:
        z = df["z"].values
    elif "zCMB" in df.columns:
        z = df["zCMB"].values
    elif "zcmb" in df.columns:
        z = df["zcmb"].values
    else:
        raise RuntimeError("SN CSV に z (または zCMB) 列が見つかりません")

    # μ, σμ 列の自動検出
    if "mu" in df.columns:
        mu = df["mu"].values
    elif "mu_SN" in df.columns:
        mu = df["mu_SN"].values
    elif "mB" in df.columns:
        # Pantheon 形式の場合（簡易版）：mB をそのまま μ とみなす
        mu = df["mB"].values
    else:
        raise RuntimeError("SN CSV に mu (または mu_SN, mB) 列が見つかりません")

    if "mu_err" in df.columns:
        mu_err = df["mu_err"].values
    elif "dmu" in df.columns:
        mu_err = df["dmu"].values
    elif "dMB" in df.columns:
        mu_err = df["dMB"].values
    else:
        raise RuntimeError("SN CSV に mu_err (または dmu, dMB) 列が見つかりません")

    # z 範囲でカット
    mask = (z >= z_min) & (z <= z_max)
    z = z[mask]
    mu = mu[mask]
    mu_err = mu_err[mask]

    return z, mu, mu_err


def load_rho_sn_profile(rho_csv):
    df = pd.read_csv(rho_csv)
    if "z" not in df.columns:
        raise RuntimeError("rho_SN CSV に z 列がありません")
    # 列名候補
    if "rho_SN" in df.columns:
        rho = df["rho_SN"].values
    elif "rho_sn" in df.columns:
        rho = df["rho_sn"].values
    else:
        raise RuntimeError("rho_SN CSV に rho_SN 列が見つかりません")

    z = df["z"].values
    # 正規化 (rho_SN(0) = 1)
    rho0 = rho[0]
    rho_norm = rho / rho0
    return z, rho_norm


def E_of_z(z, Om_r0, Om_m0, z_rho, f_sn_z):
    """
    E(z) = H(z)/H0 for given Omega_m0 and SN-mode profile f_SN(z)
    Omega_SN0 = 1 - Omega_r0 - Omega_m0 （z=0 の正規化条件）
    """
    # SN モード補間
    f_sn = np.interp(z, z_rho, f_sn_z)
    Om_SN0 = 1.0 - Om_r0 - Om_m0
    return np.sqrt(Om_r0 * (1.0 + z) ** 4 +
                   Om_m0 * (1.0 + z) ** 3 +
                   Om_SN0 * f_sn)


def make_distance_lookup(z_max, n_z, Om_r0, Om_m0, z_rho, f_sn_z, H0):
    """
    0～z_max を細かく刻んで comoving distance を積分し、
    z -> D_L(z) のルックアップテーブルを作る。
    """
    z_grid = np.linspace(0.0, z_max, n_z)
    E_grid = E_of_z(z_grid, Om_r0, Om_m0, z_rho, f_sn_z)

    # 累積台形積分で D_C/H0 を計算
    # integral_0^z dz'/E(z')
    integ = np.zeros_like(z_grid)
    for i in range(1, len(z_grid)):
        dz = z_grid[i] - z_grid[i-1]
        integ[i] = integ[i-1] + 0.5 * dz * (1.0/E_grid[i] + 1.0/E_grid[i-1])

    # D_L = (1+z) * (c/H0) * integral
    D_L = (1.0 + z_grid) * (C_LIGHT_KM_S / H0) * integ  # [Mpc] (H0 [km/s/Mpc])

    return z_grid, D_L


def mu_model_from_lookup(z_sn, z_grid, D_L_grid):
    D_L_sn = np.interp(z_sn, z_grid, D_L_grid)
    mu = 5.0 * np.log10(D_L_sn) + 25.0
    return mu


def fit_offset_C(mu_SN, mu_model, mu_err):
    """
    μ_SN = μ_model + C + noise
    に対して、最小二乗で C_best を解析的に求める。
    """
    w = 1.0 / (mu_err ** 2)
    delta = mu_SN - mu_model
    C_best = np.sum(w * delta) / np.sum(w)
    return C_best


def compute_chi2(mu_SN, mu_model, mu_err, C_best):
    res = mu_SN - (mu_model + C_best)
    chi2 = np.sum((res / mu_err) ** 2)
    return chi2, res


def main():
    parser = argparse.ArgumentParser(
        description="Phase 180: pure TFGR (matter + SN-mode) SN fit"
    )
    parser.add_argument("--sn_csv", required=True, help="SN CSV (Pantheon)")
    parser.add_argument("--rho_sn_csv", required=True,
                        help="SN-mode rho_SN(z) CSV (Phase 178 profile)")
    parser.add_argument("--H0", type=float, default=70.0,
                        help="H0 [km/s/Mpc]")
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4,
                        help="Radiation density today")
    parser.add_argument("--Om_m_min", type=float, default=0.10)
    parser.add_argument("--Om_m_max", type=float, default=0.50)
    parser.add_argument("--Om_m_steps", type=int, default=81)
    parser.add_argument("--z_min", type=float, default=0.01)
    parser.add_argument("--z_max", type=float, default=1.5)
    parser.add_argument("--n_int_z", type=int, default=800,
                        help="Integration grid size for distance")
    parser.add_argument("--out_prefix", required=True,
                        help="Prefix for output files")

    args = parser.parse_args()

    print("=== Phase 180: pure TFGR SN fit (matter + SN-mode) ===")
    print(f"SN CSV       : {args.sn_csv}")
    print(f"rho_SN CSV   : {args.rho_sn_csv}")
    print(f"H0           : {args.H0:.3f} km/s/Mpc")
    print(f"Omega_r0     : {args.Omega_r0:.5g}")
    print(f"Omega_m scan : [{args.Om_m_min:.3f}, {args.Om_m_max:.3f}] "
          f"({args.Om_m_steps} steps)")
    print(f"SN z-range   : [{args.z_min:.3f}, {args.z_max:.3f}]")
    print("===============================================")

    # --- データ読込 ---
    z_sn, mu_SN, mu_err = load_sn_csv(args.sn_csv,
                                      z_min=args.z_min,
                                      z_max=args.z_max)
    N_SN = len(z_sn)
    print(f"N_SN used    : {N_SN}")

    z_rho, f_sn = load_rho_sn_profile(args.rho_sn_csv)

    # --- Omega_m0 のグリッド探索 ---
    Om_m_grid = np.linspace(args.Om_m_min, args.Om_m_max, args.Om_m_steps)
    chi2_grid = []
    C_grid = []
    Om_SN0_grid = []

    best_idx = None
    chi2_min = np.inf

    for i, Om_m0 in enumerate(Om_m_grid):
        # SN モードの係数（z=0 の正規化条件）
        Om_SN0 = 1.0 - args.Omega_r0 - Om_m0
        Om_SN0_grid.append(Om_SN0)

        # 物理的でない（Ω_SN0 < 0）の場合はスキップして巨大な χ²
        if Om_SN0 <= 0.0:
            chi2_grid.append(np.inf)
            C_grid.append(np.nan)
            continue

        # 距離ルックアップテーブル
        z_grid, D_L_grid = make_distance_lookup(
            z_max=args.z_max,
            n_z=args.n_int_z,
            Om_r0=args.Omega_r0,
            Om_m0=Om_m0,
            z_rho=z_rho,
            f_sn_z=f_sn,
            H0=args.H0,
        )

        mu_model = mu_model_from_lookup(z_sn, z_grid, D_L_grid)
        C_best = fit_offset_C(mu_SN, mu_model, mu_err)
        chi2, _ = compute_chi2(mu_SN, mu_model, mu_err, C_best)

        chi2_grid.append(chi2)
        C_grid.append(C_best)

        if chi2 < chi2_min:
            chi2_min = chi2
            best_idx = i

    chi2_grid = np.array(chi2_grid)
    C_grid = np.array(C_grid)
    Om_SN0_grid = np.array(Om_SN0_grid)

    # --- ベスト値 ---
    if best_idx is None:
        raise RuntimeError("有効な (Omega_m0, Omega_SN0) が見つかりませんでした")

    Om_m_best = Om_m_grid[best_idx]
    Om_SN0_best = Om_SN0_grid[best_idx]
    C_best = C_grid[best_idx]

    # ベストモデルでの残差・μ_model を再計算
    z_grid_best, D_L_grid_best = make_distance_lookup(
        z_max=args.z_max,
        n_z=args.n_int_z,
        Om_r0=args.Omega_r0,
        Om_m0=Om_m_best,
        z_rho=z_rho,
        f_sn_z=f_sn,
        H0=args.H0,
    )
    mu_model_best = mu_model_from_lookup(z_sn, z_grid_best, D_L_grid_best)
    chi2_best, residuals_best = compute_chi2(mu_SN, mu_model_best, mu_err, C_best)

    dof = N_SN - 2  # Omega_m0, C をフィット
    chi2_red = chi2_best / dof

    print("------ Best-fit pure-TFGR model ------")
    print(f"Omega_m0_best  = {Om_m_best:.6f}")
    print(f"Omega_SN0_best = {Om_SN0_best:.6f}")
    print(f"C_best         = {C_best:.6f} mag")
    print(f"chi2_min       = {chi2_best:.3f}")
    print(f"dof            = {dof}")
    print(f"chi2_red       = {chi2_red:.5f}")
    print("--------------------------------------")

    # --- 結果保存 ---
    # 1) スキャンサマリ
    summary_df = pd.DataFrame({
        "Omega_m0": Om_m_grid,
        "Omega_SN0": Om_SN0_grid,
        "C_best": C_grid,
        "chi2": chi2_grid,
    })
    summary_csv = f"{args.out_prefix}_scan_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] scan summary -> {summary_csv}")

    # 2) ベストフィット残差
    best_df = pd.DataFrame({
        "z": z_sn,
        "mu_SN": mu_SN,
        "mu_err": mu_err,
        "mu_model": mu_model_best + C_best,
        "residual": mu_SN - (mu_model_best + C_best),
    })
    best_csv = f"{args.out_prefix}_best_residuals.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"[SAVE] best residuals -> {best_csv}")

    # --- 図の作成 ---
    # (a) 残差プロット
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_sn, residuals_best, yerr=mu_err, fmt=".", alpha=0.6)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("z")
    plt.ylabel(r"$\mu_{\rm SN} - \mu_{\rm model}$ [mag]")
    plt.title("Phase 180: SN residuals (pure TFGR: matter + SN-mode)")
    plt.tight_layout()
    fig1 = f"{args.out_prefix}_best_residuals.png"
    plt.savefig(fig1, dpi=150)
    print(f"[SAVE] figure -> {fig1}")

    # (b) chi2 vs Omega_m0
    plt.figure(figsize=(8, 6))
    plt.plot(Om_m_grid, chi2_grid, "o-")
    plt.axvline(Om_m_best, color="r", linestyle="--",
                label=f"best = {Om_m_best:.3f}")
    plt.xlabel(r"$\Omega_{m,0}$")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.title("Phase 180: chi^2 vs Omega_m0 (pure TFGR)")
    plt.tight_layout()
    fig2 = f"{args.out_prefix}_chi2_vs_Om_m.png"
    plt.savefig(fig2, dpi=150)
    print(f"[SAVE] figure -> {fig2}")

    print("Done.")


if __name__ == "__main__":
    main()
