#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 182: joint SN+BAO fit for LCDM vs pure-TFGR SN-mode cosmology.

Model 1 (LCDM):
  H(z)^2 = H0^2 [ Omega_r0 (1+z)^4 + Omega_m0 (1+z)^3 + Omega_L0 ]
  Omega_L0 = 1 - Omega_m0 - Omega_r0   (flat)

Model 2 (TFGR SN-mode):
  Add SN-mode effective DE with fixed w_eff_sn:
  H(z)^2 = H0^2 [ Omega_r0 (1+z)^4 + Omega_m0 (1+z)^3
                  + Omega_SN0 (1+z)^{3(1+w_eff_sn)} ]
  Omega_SN0 = 1 - Omega_m0 - Omega_r0  (flat, no separate Lambda)
  w_eff_sn is fixed from Phase 178 (e.g. -1.07).

For both models we fit:
  - Omega_m0 (scanned on a grid)
  - SN absolute magnitude offset C (nuisance, solved analytically)

We then compare chi^2, AIC, BIC for SN+BAO combined.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

C_LIGHT = 299792.458  # km/s


def load_sn_data(sn_csv):
    df = pd.read_csv(sn_csv)
    # Try to detect column names
    if "z" in df.columns:
        z = df["z"].values
    elif "z_SN" in df.columns:
        z = df["z_SN"].values
    else:
        raise RuntimeError("SN CSV: could not find 'z' column")

    if "mu" in df.columns:
        mu = df["mu"].values
    elif "mu_SN" in df.columns:
        mu = df["mu_SN"].values
    else:
        raise RuntimeError("SN CSV: could not find 'mu' or 'mu_SN' column")

    if "mu_err" in df.columns:
        mu_err = df["mu_err"].values
    elif "sigma_mu" in df.columns:
        mu_err = df["sigma_mu"].values
    else:
        raise RuntimeError("SN CSV: could not find 'mu_err' or 'sigma_mu' column")

    return z, mu, mu_err


def bao_dataset():
    """
    Simple BAO H(z) dataset (km/s/Mpc).
    Values are typical illustrative BAO measurements used in Phase 181.
    """
    z = np.array([0.24, 0.34, 0.43, 0.57, 0.60, 0.73])
    H = np.array([79.69, 83.8, 86.45, 92.9, 97.3, 97.3])
    sigma_H = np.array([2.99, 3.66, 3.68, 3.3, 2.1, 3.0])
    return z, H, sigma_H


def E_lcdm(z, Om0, Or0):
    Ol0 = 1.0 - Om0 - Or0
    return np.sqrt(Or0 * (1.0 + z) ** 4 + Om0 * (1.0 + z) ** 3 + Ol0)


def E_tfgr_sn(z, Om0, Or0, w_eff_sn):
    OmSN0 = 1.0 - Om0 - Or0
    return np.sqrt(
        Or0 * (1.0 + z) ** 4
        + Om0 * (1.0 + z) ** 3
        + OmSN0 * (1.0 + z) ** (3.0 * (1.0 + w_eff_sn))
    )


def build_mu_model(z_sn, H0, Om0, Or0, model="lcdm", w_eff_sn=-1.07):
    """
    Compute theoretical distance modulus mu_th(z) for a given cosmology.
    Uses trapezoidal integration on a fixed grid and interpolation.
    """
    z_max = float(z_sn.max()) + 0.05
    z_grid = np.linspace(0.0, z_max, 4000)

    if model == "lcdm":
        E = E_lcdm(z_grid, Om0, Or0)
    elif model == "tfgr":
        E = E_tfgr_sn(z_grid, Om0, Or0, w_eff_sn)
    else:
        raise ValueError("Unknown model: {}".format(model))

    f = 1.0 / E
    integral = np.zeros_like(z_grid)
    # trapezoidal cumulative integral of f(z)
    for i in range(1, len(z_grid)):
        dz = z_grid[i] - z_grid[i - 1]
        integral[i] = integral[i - 1] + 0.5 * (f[i] + f[i - 1]) * dz

    # comoving distance D_C(z)
    D_C = (C_LIGHT / H0) * integral  # in Mpc

    # luminosity distance D_L(z) = (1+z) D_C(z)
    D_L = (1.0 + z_grid) * D_C

    # interpolate to SN redshifts
    D_L_sn = np.interp(z_sn, z_grid, D_L)

    # distance modulus
    mu_th = 5.0 * np.log10(D_L_sn) + 25.0
    return mu_th


def best_C_and_chi2(mu_data, mu_err, mu_model):
    """
    For fixed mu_model(z), minimize chi^2 in C for mu_model + C.
    Returns C_best and chi^2_min.
    """
    w = 1.0 / (mu_err ** 2)
    delta = mu_data - mu_model
    C_best = np.sum(w * delta) / np.sum(w)
    chi2 = np.sum(((delta - C_best) ** 2) * w)
    return C_best, chi2


def chi2_bao(H0, Om0, Or0, model="lcdm", w_eff_sn=-1.07):
    z_bao, H_data, H_err = bao_dataset()
    if model == "lcdm":
        E = E_lcdm(z_bao, Om0, Or0)
    elif model == "tfgr":
        E = E_tfgr_sn(z_bao, Om0, Or0, w_eff_sn)
    else:
        raise ValueError("Unknown model")

    H_model = H0 * E
    chi2 = np.sum(((H_model - H_data) / H_err) ** 2)
    return chi2


def main():
    parser = argparse.ArgumentParser(
        description="Phase 182: joint SN+BAO fit for LCDM vs pure TFGR SN-mode cosmology"
    )
    parser.add_argument("--sn_csv", required=True, help="SN CSV file (Pantheon-like)")
    parser.add_argument("--H0", type=float, default=70.0, help="Hubble constant [km/s/Mpc]")
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4, help="Radiation density today")
    parser.add_argument("--Om_m_min", type=float, default=0.10, help="Min Omega_m0")
    parser.add_argument("--Om_m_max", type=float, default=0.50, help="Max Omega_m0")
    parser.add_argument("--Om_m_steps", type=int, default=41, help="Number of Omega_m0 steps")
    parser.add_argument(
        "--w_eff_sn",
        type=float,
        default=-1.07,
        help="Effective w for TFGR SN-mode (from Phase 178)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="phase182_joint_SN_BAO",
        help="Output prefix",
    )
    args = parser.parse_args()

    sn_csv = args.sn_csv
    H0 = args.H0
    Or0 = args.Omega_r0
    w_eff_sn = args.w_eff_sn
    out_prefix = args.out_prefix

    print("=== Phase 182: joint SN+BAO fit (LCDM vs TFGR SN-mode) ===")
    print(f"SN CSV   : {sn_csv}")
    print(f"H0       : {H0:.2f}")
    print(f"Omega_r0 : {Or0:.5g}")
    print(f"w_eff_sn : {w_eff_sn:.3f}")
    print("==============================================")

    # Load SN data
    z_sn, mu_sn, mu_err = load_sn_data(sn_csv)
    N_SN = len(z_sn)
    z_bao, _, _ = bao_dataset()
    N_BAO = len(z_bao)
    N_tot = N_SN + N_BAO

    # Omega_m0 grid
    Om_grid = np.linspace(args.Om_m_min, args.Om_m_max, args.Om_m_steps)

    # Containers
    results_lcdm = []
    results_tfgr = []

    # --- Scan LCDM ---
    for Om in Om_grid:
        mu_model = build_mu_model(z_sn, H0, Om, Or0, model="lcdm")
        C_best, chi2_sn = best_C_and_chi2(mu_sn, mu_err, mu_model)
        chi2_b = chi2_bao(H0, Om, Or0, model="lcdm")
        chi2_tot = chi2_sn + chi2_b
        results_lcdm.append((Om, C_best, chi2_sn, chi2_b, chi2_tot))

    # --- Scan TFGR SN-mode ---
    for Om in Om_grid:
        mu_model = build_mu_model(z_sn, H0, Om, Or0, model="tfgr", w_eff_sn=w_eff_sn)
        C_best, chi2_sn = best_C_and_chi2(mu_sn, mu_err, mu_model)
        chi2_b = chi2_bao(H0, Om, Or0, model="tfgr", w_eff_sn=w_eff_sn)
        chi2_tot = chi2_sn + chi2_b
        results_tfgr.append((Om, C_best, chi2_sn, chi2_b, chi2_tot))

    # Convert to DataFrame and save
    df_lcdm = pd.DataFrame(
        results_lcdm,
        columns=["Omega_m0", "C_best", "chi2_SN", "chi2_BAO", "chi2_tot"],
    )
    df_tfgr = pd.DataFrame(
        results_tfgr,
        columns=["Omega_m0", "C_best", "chi2_SN", "chi2_BAO", "chi2_tot"],
    )

    df_lcdm.to_csv(out_prefix + "_lcdm_scan.csv", index=False)
    df_tfgr.to_csv(out_prefix + "_tfgr_scan.csv", index=False)

    # Find best fits
    idx_l = int(np.argmin(df_lcdm["chi2_tot"].values))
    idx_t = int(np.argmin(df_tfgr["chi2_tot"].values))

    best_l = df_lcdm.iloc[idx_l]
    best_t = df_tfgr.iloc[idx_t]

    k_params = 2  # Omega_m0 + C for both models
    chi2_l = best_l["chi2_tot"]
    chi2_t = best_t["chi2_tot"]

    AIC_l = chi2_l + 2 * k_params
    AIC_t = chi2_t + 2 * k_params
    BIC_l = chi2_l + k_params * np.log(N_tot)
    BIC_t = chi2_t + k_params * np.log(N_tot)

    print("------ Best-fit LCDM (SN+BAO) ------")
    print(f"Omega_m0_best = {best_l['Omega_m0']:.5f}")
    print(f"Omega_L0_best = {1.0 - best_l['Omega_m0'] - Or0:.5f}")
    print(f"C_best        = {best_l['C_best']:.6f} mag")
    print(f"chi2_SN       = {best_l['chi2_SN']:.3f}")
    print(f"chi2_BAO      = {best_l['chi2_BAO']:.3f}")
    print(f"chi2_tot      = {chi2_l:.3f}")
    print(f"AIC           = {AIC_l:.3f}")
    print(f"BIC           = {BIC_l:.3f}")
    print("------------------------------------")

    print("------ Best-fit TFGR SN-mode (SN+BAO) ------")
    print(f"Omega_m0_best  = {best_t['Omega_m0']:.5f}")
    print(f"Omega_SN0_best = {1.0 - best_t['Omega_m0'] - Or0:.5f}")
    print(f"w_eff_sn       = {w_eff_sn:.3f}")
    print(f"C_best         = {best_t['C_best']:.6f} mag")
    print(f"chi2_SN        = {best_t['chi2_SN']:.3f}")
    print(f"chi2_BAO       = {best_t['chi2_BAO']:.3f}")
    print(f"chi2_tot       = {chi2_t:.3f}")
    print(f"AIC            = {AIC_t:.3f}")
    print(f"BIC            = {BIC_t:.3f}")
    print("-------------------------------------------")

    print("------ Differences (TFGR - LCDM) ------")
    print(f"Delta chi2 = {chi2_t - chi2_l:.3f}")
    print(f"Delta AIC  = {AIC_t - AIC_l:.3f}")
    print(f"Delta BIC  = {BIC_t - BIC_l:.3f}")
    print("---------------------------------------")

    # Save best-fit residuals for both models
    # LCDM residuals
    mu_model_l = build_mu_model(z_sn, H0, best_l["Omega_m0"], Or0, model="lcdm")
    C_l = best_l["C_best"]
    res_l = mu_sn - (mu_model_l + C_l)
    df_res_l = pd.DataFrame(
        {
            "z": z_sn,
            "mu_SN": mu_sn,
            "mu_err": mu_err,
            "mu_model": mu_model_l + C_l,
            "residual": res_l,
        }
    )
    df_res_l.to_csv(out_prefix + "_lcdm_best_residuals.csv", index=False)

    # TFGR residuals
    mu_model_t = build_mu_model(
        z_sn, H0, best_t["Omega_m0"], Or0, model="tfgr", w_eff_sn=w_eff_sn
    )
    C_t = best_t["C_best"]
    res_t = mu_sn - (mu_model_t + C_t)
    df_res_t = pd.DataFrame(
        {
            "z": z_sn,
            "mu_SN": mu_sn,
            "mu_err": mu_err,
            "mu_model": mu_model_t + C_t,
            "residual": res_t,
        }
    )
    df_res_t.to_csv(out_prefix + "_tfgr_best_residuals.csv", index=False)

    # Plot chi2 vs Omega_m0 for both models
    plt.figure(figsize=(8, 6))
    plt.plot(df_lcdm["Omega_m0"], df_lcdm["chi2_tot"], "o-", label="LCDM")
    plt.plot(df_tfgr["Omega_m0"], df_tfgr["chi2_tot"], "s-", label="TFGR SN-mode")
    plt.axvline(best_l["Omega_m0"], color="C0", linestyle="--", alpha=0.5)
    plt.axvline(best_t["Omega_m0"], color="C1", linestyle="--", alpha=0.5)
    plt.xlabel(r"$\Omega_{m,0}$")
    plt.ylabel(r"$\chi^2_{\mathrm{SN+BAO}}$")
    plt.title("Phase 182: chi^2 vs Omega_m0 (SN+BAO joint)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_chi2_vs_Om_m.png", dpi=150)
    plt.close()

    print(f"[Phase 182] Saved scan CSVs and chi2 plot with prefix: {out_prefix}")


if __name__ == "__main__":
    main()
