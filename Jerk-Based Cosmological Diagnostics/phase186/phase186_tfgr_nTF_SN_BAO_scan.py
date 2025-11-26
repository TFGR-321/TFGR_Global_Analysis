#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 186: TFGR power-law dark energy (n_TF) scan with SN + BAO/H(z)

- Flat universe: Om + Or + OTF = 1
- TFGR component: rho_TF(z) ∝ (1+z)^{n_TF}
- w_TF = -1 + n_TF/3
- H(z)^2 = H0^2 [ Or0 (1+z)^4 + Om0 (1+z)^3 + OTF0 (1+z)^{n_TF} ]

We scan over (Om0, n_TF), and for each pair:
- compute SN chi2 with analytic best-fit C (magnitude offset)
- compute BAO chi2 on H(z)
- combine chi2_tot = chi2_SN + chi2_BAO

Outputs:
- scan CSV with all grid points
- best-fit parameters and chi2 summary
"""

import argparse
import numpy as np
import pandas as pd

# 光速 [km/s]
C_LIGHT = 299792.458


def read_sn(sn_csv):
    """Pantheon-like SN: columns z, mu, mu_err"""
    df = pd.read_csv(sn_csv)
    for col in ["z", "mu", "mu_err"]:
        if col not in df.columns:
            raise RuntimeError(f"SN CSV に '{col}' 列がありません: {sn_csv}")
    return df["z"].values, df["mu"].values, df["mu_err"].values


def read_bao(bao_csv):
    """BAO/H(z) data: columns z, H_obs, H_err"""
    df = pd.read_csv(bao_csv)
    for col in ["z", "H_obs", "H_err"]:
        if col not in df.columns:
            raise RuntimeError(f"BAO CSV に '{col}' 列がありません: {bao_csv}")
    return df["z"].values, df["H_obs"].values, df["H_err"].values


def E_TFGR(z, Om0, Or0, n_TF):
    """
    E(z) = H(z)/H0 for flat (Om + Or + OTF = 1) with TFGR power-law component.
    """
    z = np.asarray(z)
    OTF0 = 1.0 - Om0 - Or0
    if OTF0 < 0:
        # unphysical, return NaN so that chi2 becomes huge
        return np.full_like(z, np.nan)
    term_r = Or0 * (1.0 + z) ** 4
    term_m = Om0 * (1.0 + z) ** 3
    term_tf = OTF0 * (1.0 + z) ** n_TF
    E2 = term_r + term_m + term_tf
    # avoid negative under small numerical fluctuations
    E2 = np.where(E2 > 0.0, E2, np.nan)
    return np.sqrt(E2)


def H_TFGR(z, H0, Om0, Or0, n_TF):
    return H0 * E_TFGR(z, Om0, Or0, n_TF)


def comoving_distance(z_arr, H0, Om0, Or0, n_TF, n_int=200):
    """
    Comoving distance D_C(z) = c/H0 ∫_0^z dz'/E(z')
    数値積分: 台形則 (numpy.trapz) でシンプルに。
    z_arr は 1D array。
    """
    z_arr = np.asarray(z_arr)
    Dc = np.zeros_like(z_arr, dtype=float)

    for i, z in enumerate(z_arr):
        if z <= 0:
            Dc[i] = 0.0
            continue
        # integrate from 0 to z with n_int steps
        zz = np.linspace(0.0, z, n_int)
        Ez = E_TFGR(zz, Om0, Or0, n_TF)
        if np.any(np.isnan(Ez)):
            Dc[i] = np.nan
            continue
        integrand = 1.0 / Ez
        integral = np.trapz(integrand, zz)
        Dc[i] = (C_LIGHT / H0) * integral  # [Mpc]
    return Dc


def distance_modulus_TFGR(z_sn, H0, Om0, Or0, n_TF):
    """
    TFGR power-law modelの距離モジュラス μ_model(z)
    """
    Dc = comoving_distance(z_sn, H0, Om0, Or0, n_TF, n_int=400)
    if np.any(np.isnan(Dc)):
        return np.full_like(z_sn, np.nan)
    Dl = (1.0 + z_sn) * Dc  # [Mpc]
    mu_model = 5.0 * np.log10(Dl) + 25.0
    return mu_model


def chi2_SN_with_C(z_sn, mu_sn, mu_err, H0, Om0, Or0, n_TF):
    """
    SN χ² を計算（C を解析的に最適化）
    mu_model(z) に C を足してフィット。
    """
    mu_model = distance_modulus_TFGR(z_sn, H0, Om0, Or0, n_TF)
    if np.any(np.isnan(mu_model)):
        return np.inf, np.nan

    w = 1.0 / (mu_err ** 2)
    # best-fit C
    num = np.sum(w * (mu_sn - mu_model))
    den = np.sum(w)
    C_best = num / den
    # chi2_min
    resid = mu_sn - (mu_model + C_best)
    chi2 = np.sum(w * resid ** 2)
    return chi2, C_best


def chi2_BAO(z_bao, H_obs, H_err, H0, Om0, Or0, n_TF):
    H_model = H_TFGR(z_bao, H0, Om0, Or0, n_TF)
    if np.any(np.isnan(H_model)):
        return np.inf
    chi2 = np.sum(((H_obs - H_model) / H_err) ** 2)
    return chi2


def main():
    parser = argparse.ArgumentParser(
        description="Phase 186: TFGR n_TF scan with SN+BAO"
    )
    parser.add_argument("--sn_csv", type=str, required=True,
                        help="Pantheon SN CSV (z,mu,mu_err)")
    parser.add_argument("--bao_csv", type=str, required=True,
                        help="BAO H(z) CSV (z,H_obs,H_err)")
    parser.add_argument("--H0", type=float, default=70.0,
                        help="H0 [km/s/Mpc]")
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4,
                        help="Radiation density parameter today")
    parser.add_argument("--Om_m_min", type=float, default=0.2)
    parser.add_argument("--Om_m_max", type=float, default=0.4)
    parser.add_argument("--Om_m_steps", type=int, default=21)
    parser.add_argument("--nTF_min", type=float, default=-0.6)
    parser.add_argument("--nTF_max", type=float, default=0.2)
    parser.add_argument("--nTF_steps", type=int, default=33)
    parser.add_argument("--z_min_SN", type=float, default=0.01)
    parser.add_argument("--z_max_SN", type=float, default=1.5)
    parser.add_argument("--out_prefix", type=str,
                        default="phase186_tfgr_nTF_scan")
    args = parser.parse_args()

    # --- load data ---
    z_sn, mu_sn, mu_err = read_sn(args.sn_csv)
    z_bao, H_obs, H_err = read_bao(args.bao_csv)

    # SN z-range cut
    sn_mask = (z_sn >= args.z_min_SN) & (z_sn <= args.z_max_SN)
    z_sn_use = z_sn[sn_mask]
    mu_sn_use = mu_sn[sn_mask]
    mu_err_use = mu_err[sn_mask]
    N_SN = len(z_sn_use)

    print("=== Phase 186: TFGR n_TF scan (SN+BAO) ===")
    print(f"SN file     : {args.sn_csv}")
    print(f"BAO file    : {args.bao_csv}")
    print(f"H0          : {args.H0:.3f}")
    print(f"Omega_r0    : {args.Omega_r0:.6e}")
    print(f"SN z-range  : [{args.z_min_SN}, {args.z_max_SN}] (N_SN={N_SN})")
    print(f"Om_m range  : [{args.Om_m_min}, {args.Om_m_max}] steps={args.Om_m_steps}")
    print(f"n_TF range  : [{args.nTF_min}, {args.nTF_max}] steps={args.nTF_steps}")
    print(f"out_prefix  : {args.out_prefix}")
    print("==========================================")

    Om_grid = np.linspace(args.Om_m_min, args.Om_m_max, args.Om_m_steps)
    nTF_grid = np.linspace(args.nTF_min, args.nTF_max, args.nTF_steps)

    records = []
    best_rec = None
    best_chi2_tot = np.inf

    dof_SN = N_SN - 1  # minus C
    # total parameters: (Om0, n_TF, C) = 3
    # dof_tot = N_SN + N_BAO - 3 だが、ここでは chi2_red だけ参照

    for Om0 in Om_grid:
        for nTF in nTF_grid:
            # skip unphysical
            if 1.0 - Om0 - args.Omega_r0 <= 0.0:
                continue

            chi2_sn, C_best = chi2_SN_with_C(
                z_sn_use, mu_sn_use, mu_err_use,
                args.H0, Om0, args.Omega_r0, nTF
            )
            if not np.isfinite(chi2_sn):
                continue

            chi2_bao = chi2_BAO(
                z_bao, H_obs, H_err,
                args.H0, Om0, args.Omega_r0, nTF
            )
            if not np.isfinite(chi2_bao):
                continue

            chi2_tot = chi2_sn + chi2_bao
            # dof_tot: SN: N_SN -1(C), BAO: N_BAO, minus Om0,nTF(2) → N_SN + N_BAO -3
            dof_tot = (N_SN - 1) + len(z_bao) - 2
            chi2_red = chi2_tot / dof_tot

            rec = dict(
                Om0=Om0,
                n_TF=nTF,
                Omega_TF0=1.0 - Om0 - args.Omega_r0,
                H0=args.H0,
                chi2_SN=chi2_sn,
                chi2_BAO=chi2_bao,
                chi2_tot=chi2_tot,
                dof_tot=dof_tot,
                chi2_red=chi2_red,
                C_best=C_best,
            )
            records.append(rec)

            if chi2_tot < best_chi2_tot:
                best_chi2_tot = chi2_tot
                best_rec = rec

    df_scan = pd.DataFrame(records)
    scan_csv = args.out_prefix + "_scan.csv"
    df_scan.to_csv(scan_csv, index=False)
    print(f"[Phase186] Saved scan CSV -> {scan_csv}")

    if best_rec is not None:
        print("------ Best-fit (TFGR n_TF power-law, SN+BAO) ------")
        print(f"Omega_m0_best  = {best_rec['Om0']:.5f}")
        print(f"Omega_TF0_best = {best_rec['Omega_TF0']:.5f}")
        print(f"n_TF_best      = {best_rec['n_TF']:.5f}")
        print(f"H0             = {best_rec['H0']:.3f}")
        print(f"C_best         = {best_rec['C_best']:.5f} mag")
        print(f"chi2_SN        = {best_rec['chi2_SN']:.3f}")
        print(f"chi2_BAO       = {best_rec['chi2_BAO']:.3f}")
        print(f"chi2_tot       = {best_rec['chi2_tot']:.3f}")
        print(f"dof_tot        = {best_rec['dof_tot']:.0f}")
        print(f"chi2_red       = {best_rec['chi2_red']:.3f}")
        print("---------------------------------------------------")

        best_csv = args.out_prefix + "_best_summary.csv"
        pd.DataFrame([best_rec]).to_csv(best_csv, index=False)
        print(f"[Phase186] Saved best-fit summary -> {best_csv}")
    else:
        print("[Phase186] No valid model found (all OTF0<0 or numerical issues).")


if __name__ == "__main__":
    main()
