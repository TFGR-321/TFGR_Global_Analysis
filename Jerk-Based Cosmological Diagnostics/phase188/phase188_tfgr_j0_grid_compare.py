#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 188: ΛCDM vs TFGR power-law (n_TF) joint SN+BAO comparison
and j(0) prediction for TFGR.

Usage example:
python phase188_tfgr_j0_grid_compare.py \
  --sn_csv pantheon_SN.csv \
  --bao_csv bao_Hz_data.csv \
  --H0 70.0 \
  --Omega_r0 1.0e-4 \
  --Om_m_min 0.20 --Om_m_max 0.40 --Om_m_steps 41 \
  --nTF_min -0.8 --nTF_max -0.2 --nTF_steps 31 \
  --out_prefix phase188_tfgr_j0_grid
"""

import argparse
import numpy as np
import pandas as pd
from math import sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt

C_LIGHT = 299792.458  # km/s


# -----------------------------
# Cosmology helpers
# -----------------------------
def E_LCDM(z, Om_m0, Om_r0):
    Om_L0 = 1.0 - Om_m0 - Om_r0
    return np.sqrt(Om_r0 * (1 + z) ** 4 + Om_m0 * (1 + z) ** 3 + Om_L0)


def E_TFGR(z, Om_m0, Om_r0, n_TF):
    Om_TF0 = 1.0 - Om_m0 - Om_r0
    return np.sqrt(Om_r0 * (1 + z) ** 4 +
                   Om_m0 * (1 + z) ** 3 +
                   Om_TF0 * (1 + z) ** n_TF)


def luminosity_distance(z, H0, E_func, *E_args):
    """Flat universe: d_L = (1+z) c/H0 ∫ dz'/E(z')."""
    def integrand(zp):
        return 1.0 / E_func(zp, *E_args)

    val, _ = quad(integrand, 0.0, z, epsabs=1e-6, epsrel=1e-6)
    d_c = (C_LIGHT / H0) * val  # Mpc
    d_l = (1.0 + z) * d_c
    return d_l


def distance_modulus(z, H0, E_func, *E_args):
    d_l = luminosity_distance(z, H0, E_func, *E_args)
    return 5.0 * (np.log10(d_l) + 5.0)  # μ = 5 log10(d_L/Mpc) + 25


# -----------------------------
# χ² for SN with analytic C_best
# -----------------------------
def chi2_SN(mu_SN, sig_mu, mu_th):
    """
    χ²_SN(μ_SN, μ_th, σ) minimized over C (absolute magnitude offset).
    C_best = Σ(μ_SN - μ_th)/σ² / Σ(1/σ²)
    """
    w = 1.0 / (sig_mu ** 2)
    delta = mu_SN - mu_th
    C_best = np.sum(w * delta) / np.sum(w)   # analytic optimum
    res = delta - C_best
    chi2 = np.sum(w * res * res)
    return chi2, C_best


# -----------------------------
# χ² for BAO H(z)
# -----------------------------
def chi2_BAO(z_bao, H_obs, H_err, H0, E_func, *E_args):
    H_model = H0 * E_func(z_bao, *E_args)
    chi2 = np.sum(((H_obs - H_model) / H_err) ** 2)
    return chi2


# -----------------------------
# j(0) for TFGR model
# -----------------------------
def compute_j0_TFGR(H0, Om_m0, Om_r0, n_TF):
    """
    数値微分から TFGR power-law の j(0) を計算。
    z ∈ [-0.01, 0.01] の細かいグリッドで H(z) → q(z) → j(z)。
    """

    def H_of_z(z):
        return H0 * E_TFGR(z, Om_m0, Om_r0, n_TF)

    z_arr = np.linspace(-0.01, 0.01, 81)
    H_arr = np.array([H_of_z(z) for z in z_arr])

    # 1階・2階微分
    dH_dz = np.gradient(H_arr, z_arr)
    q_arr = (1.0 + z_arr) / H_arr * dH_dz - 1.0
    dq_dz = np.gradient(q_arr, z_arr)
    j_arr = q_arr * (2.0 * q_arr + 1.0) + (1.0 + z_arr) * dq_dz

    # z=0 に最も近い点
    idx0 = np.argmin(np.abs(z_arr))
    return j_arr[idx0]


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sn_csv", type=str, required=True)
    parser.add_argument("--bao_csv", type=str, required=True)
    parser.add_argument("--H0", type=float, default=70.0)
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4)

    parser.add_argument("--Om_m_min", type=float, default=0.2)
    parser.add_argument("--Om_m_max", type=float, default=0.4)
    parser.add_argument("--Om_m_steps", type=int, default=41)

    parser.add_argument("--nTF_min", type=float, default=-0.8)
    parser.add_argument("--nTF_max", type=float, default=-0.2)
    parser.add_argument("--nTF_steps", type=int, default=31)

    parser.add_argument("--z_min_SN", type=float, default=0.01)
    parser.add_argument("--z_max_SN", type=float, default=1.5)

    parser.add_argument("--out_prefix", type=str,
                        default="phase188_tfgr_j0_grid")
    args = parser.parse_args()

    H0 = args.H0
    Om_r0 = args.Omega_r0

    # ---------------- SN data ----------------
    sn = pd.read_csv(args.sn_csv)
    # カラム名はあなたのファイルに合わせて調整してください
    # ここではフェーズ182と同じ想定:
    # z, mu_SN, mu_err
    z_sn = sn["z"].values
    mu_sn = sn["mu_SN"].values
    sig_mu = sn["mu_err"].values

    # z 範囲でフィルタ
    mask_sn = (z_sn >= args.z_min_SN) & (z_sn <= args.z_max_SN)
    z_sn = z_sn[mask_sn]
    mu_sn = mu_sn[mask_sn]
    sig_mu = sig_mu[mask_sn]
    N_SN = len(z_sn)

    # ---------------- BAO data ----------------
    bao = pd.read_csv(args.bao_csv)
    # カラム名: z, Hz, Hz_err （さきほど作った形式）
    z_bao = bao["z"].values
    H_obs = bao["Hz"].values
    H_err = bao["Hz_err"].values
    N_BAO = len(z_bao)

    print("========================================")
    print("Phase 188: ΛCDM vs TFGR(n_TF) SN+BAO grid")
    print("----------------------------------------")
    print(f"H0       = {H0:.3f}")
    print(f"Omega_r0 = {Om_r0:.4e}")
    print(f"N_SN     = {N_SN}")
    print(f"N_BAO    = {N_BAO}")
    print("========================================")

    # パラメータグリッド
    Om_m_grid = np.linspace(args.Om_m_min, args.Om_m_max, args.Om_m_steps)
    nTF_grid = np.linspace(args.nTF_min, args.nTF_max, args.nTF_steps)

    # 結果をためるリスト
    rows_lcdm = []
    rows_tfgr = []

    # ---------------- ΛCDM scan (1D: Om_m) ----------------
    best_lcdm = None  # (chi2_tot, Om_m0_best, C_best, chi2_SN, chi2_BAO)

    for Om_m0 in Om_m_grid:
        # SN
        mu_th_sn = np.array(
            [distance_modulus(z, H0, E_LCDM, Om_m0, Om_r0) for z in z_sn]
        )
        chi2_sn, C_best = chi2_SN(mu_sn, sig_mu, mu_th_sn)

        # BAO
        chi2_bao = chi2_BAO(z_bao, H_obs, H_err, H0, E_LCDM, Om_m0, Om_r0)

        chi2_tot = chi2_sn + chi2_bao
        dof = N_SN + N_BAO - 2  # Om_m0 + C の2パラメータ

        rows_lcdm.append(
            dict(model="LCDM",
                 Om_m0=Om_m0,
                 n_TF=np.nan,
                 C_best=C_best,
                 chi2_SN=chi2_sn,
                 chi2_BAO=chi2_bao,
                 chi2_tot=chi2_tot,
                 dof=dof)
        )

        if (best_lcdm is None) or (chi2_tot < best_lcdm[0]):
            best_lcdm = (chi2_tot, Om_m0, C_best, chi2_sn, chi2_bao, dof)

    # ---------------- TFGR scan (2D: Om_m, n_TF) ----------------
    best_tfgr = None  # (chi2_tot, Om_m0, n_TF, C_best, chi2_SN, chi2_BAO, dof)

    for Om_m0 in Om_m_grid:
        for n_TF in nTF_grid:
            # TF 成分の Ω が負にならないよう一応チェック
            Om_TF0 = 1.0 - Om_m0 - Om_r0
            if Om_TF0 <= 0.0:
                continue

            # SN
            mu_th_sn = np.array(
                [distance_modulus(z, H0, E_TFGR, Om_m0, Om_r0, n_TF)
                 for z in z_sn]
            )
            chi2_sn, C_best = chi2_SN(mu_sn, sig_mu, mu_th_sn)

            # BAO
            chi2_bao = chi2_BAO(z_bao, H_obs, H_err, H0,
                                E_TFGR, Om_m0, Om_r0, n_TF)

            chi2_tot = chi2_sn + chi2_bao
            # パラメータ: Om_m0, n_TF, C → 3
            dof = N_SN + N_BAO - 3

            rows_tfgr.append(
                dict(model="TFGR",
                     Om_m0=Om_m0,
                     n_TF=n_TF,
                     C_best=C_best,
                     chi2_SN=chi2_sn,
                     chi2_BAO=chi2_bao,
                     chi2_tot=chi2_tot,
                     dof=dof)
            )

            if (best_tfgr is None) or (chi2_tot < best_tfgr[0]):
                best_tfgr = (chi2_tot, Om_m0, n_TF, C_best,
                             chi2_sn, chi2_bao, dof)

    # ---------------- 結果整理 ----------------
    df_lcdm = pd.DataFrame(rows_lcdm)
    df_tfgr = pd.DataFrame(rows_tfgr)

    df_lcdm.to_csv(args.out_prefix + "_lcdm_scan.csv", index=False)
    df_tfgr.to_csv(args.out_prefix + "_tfgr_scan.csv", index=False)

    # ベスト値取り出し
    chi2_l, Om_m_l, C_l, chi2_sn_l, chi2_bao_l, dof_l = best_lcdm
    chi2_t, Om_m_t, nTF_t, C_t, chi2_sn_t, chi2_bao_t, dof_t = best_tfgr

    # AIC/BIC
    k_l = 2  # Om_m0, C
    k_t = 3  # Om_m0, n_TF, C

    AIC_l = chi2_l + 2 * k_l
    BIC_l = chi2_l + k_l * np.log(N_SN + N_BAO)

    AIC_t = chi2_t + 2 * k_t
    BIC_t = chi2_t + k_t * np.log(N_SN + N_BAO)

    # TFGR の j(0)
    j0_tfgr = compute_j0_TFGR(H0, Om_m_t, Om_r0, nTF_t)

    print("\n====== Best-fit ΛCDM (SN+BAO) ======")
    print(f"Omega_m0_best = {Om_m_l:.5f}")
    print(f"C_best        = {C_l:.5f} mag")
    print(f"chi2_SN       = {chi2_sn_l:.3f}")
    print(f"chi2_BAO      = {chi2_bao_l:.3f}")
    print(f"chi2_tot      = {chi2_l:.3f}")
    print(f"dof           = {dof_l}")
    print(f"chi2_red      = {chi2_l/dof_l:.5f}")
    print(f"AIC           = {AIC_l:.3f}")
    print(f"BIC           = {BIC_l:.3f}")

    print("\n====== Best-fit TFGR power-law (SN+BAO) ======")
    print(f"Omega_m0_best = {Om_m_t:.5f}")
    print(f"n_TF_best     = {nTF_t:.5f}")
    print(f"Omega_TF0     = {1.0-Om_m_t-Om_r0:.5f}")
    print(f"C_best        = {C_t:.5f} mag")
    print(f"chi2_SN       = {chi2_sn_t:.3f}")
    print(f"chi2_BAO      = {chi2_bao_t:.3f}")
    print(f"chi2_tot      = {chi2_t:.3f}")
    print(f"dof           = {dof_t}")
    print(f"chi2_red      = {chi2_t/dof_t:.5f}")
    print(f"AIC           = {AIC_t:.3f}")
    print(f"BIC           = {BIC_t:.3f}")
    print(f"j_TFGR(0)     = {j0_tfgr:.4f}")
    print("  (ΛCDM の場合は常に j(0)=1)")

    print("\n====== Differences (TFGR - ΛCDM) ======")
    print(f"Δchi2 = {chi2_t - chi2_l:+.3f}")
    print(f"ΔAIC  = {AIC_t - AIC_l:+.3f}")
    print(f"ΔBIC  = {BIC_t - BIC_l:+.3f}")
    print("========================================")

    # おまけ: χ² vs Ωm の簡単な図（ΛCDM と TFGR best-nTF を比較）
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(df_lcdm["Om_m0"], df_lcdm["chi2_tot"],
                 "o-", label="ΛCDM")
        # TFGR のうち best_nTF 付近だけ抽出
        df_tfgr_bestn = df_tfgr.loc[np.isclose(df_tfgr["n_TF"], nTF_t)]
        if len(df_tfgr_bestn) > 0:
            plt.plot(df_tfgr_bestn["Om_m0"], df_tfgr_bestn["chi2_tot"],
                     "s-", label=f"TFGR (n_TF={nTF_t:.2f})")
        plt.xlabel(r"$\Omega_{m,0}$")
        plt.ylabel(r"$\chi^2_{\rm SN+BAO}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_prefix + "_chi2_vs_Om_m.png", dpi=150)
        plt.close()
    except Exception as e:
        print("[WARN] Plotting failed:", e)


if __name__ == "__main__":
    main()
