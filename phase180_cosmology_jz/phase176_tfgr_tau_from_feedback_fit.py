#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 176:
TFGR feedback potential → theoretical tau(z) → SN Δμ(z) fit
"""

import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Load TFGR feedback profile (Phi_fb(z))
# ---------------------------------------------------------------

def load_tfgr_feedback(csv_file):
    df = pd.read_csv(csv_file)
    # Column names may vary; try typical names
    cand_cols = ["z","Phi_feedback","Phi_fb","Phi_fb_z","Phi_fb_val"]
    z_col = None
    fb_col = None

    for c in df.columns:
        if c.lower()=="z":
            z_col = c
        if "phi" in c.lower() and ("fb" in c.lower() or "feedback" in c.lower()):
            fb_col = c

    if z_col is None or fb_col is None:
        raise ValueError("Cannot find z or Phi_fb column in TFGR file.")

    z = df[z_col].to_numpy()
    phi = df[fb_col].to_numpy()
    return z, phi

# ---------------------------------------------------------------
# 2. Compute theoretical tau(z)
#    tau(z) = ∫ phi_fb(z')/(1+z') dz'
# ---------------------------------------------------------------

def compute_tau(z_grid, phi_grid):
    spline_phi = UnivariateSpline(z_grid, phi_grid, s=0)
    z_fine = np.linspace(0, np.max(z_grid), 2000)
    phi_fine = spline_phi(z_fine)
    integrand = phi_fine / (1.0 + z_fine)
    tau_fine = cumtrapz(integrand, z_fine, initial=0.0)
    return z_fine, tau_fine

# ---------------------------------------------------------------
# 3. Convert tau(z) → Δμ(z)
#    Δμ_model(z) = -k * tau(z)    (k determined by SN fit)
# ---------------------------------------------------------------

# Compute SN model chi^2 for given k
def chi2_SN(k, z_SN, mu_SN, mu_err, spline_tau):
    # theoretical delta-mu
    tau_SN = spline_tau(z_SN)
    dmu_model = -k * tau_SN
    # model for mu = mu_TFGR + dmu
    # But TFGR mu baseline = 5 log10( D_TFGR ) + C
    # We absorb mu_TFGR + C into (mu_SN - dmu)
    # → compute residual: mu_SN - (mu_TFGR + C) = dmu
    # So chi2 is based on dmu
    # Actually we fit residual = mu_SN - mu_TFGR - C_best:
    # here we assume residuals = dmu_model
    # So chi2 = Σ ((dmu_SN - dmu_model)/err)^2
    # We measure dmu_SN = mu_SN - mu_TFGR - C_best.
    # But C_best is unknown here — we remove mean.
    # Instead, subtract mean from both SN and model.
    dmu_SN = mu_SN - np.mean(mu_SN)
    dmu_model = dmu_model - np.mean(dmu_model)
    res = (dmu_SN - dmu_model) / mu_err
    return np.sum(res*res)

# ---------------------------------------------------------------
# 4. Fit k by scanning
# ---------------------------------------------------------------

def fit_k(z_fine, tau_fine, z_SN, mu_SN, mu_err):
    spline_tau = UnivariateSpline(z_fine, tau_fine, s=0)

    k_vals = np.linspace(0, 5000, 600)  # wide search
    chi2_vals = []

    for k in k_vals:
        chi2 = chi2_SN(k, z_SN, mu_SN, mu_err, spline_tau)
        chi2_vals.append(chi2)

    chi2_vals = np.array(chi2_vals)
    idx = np.argmin(chi2_vals)

    return k_vals[idx], chi2_vals[idx], k_vals, chi2_vals, spline_tau

# ---------------------------------------------------------------
# 5. Main routine
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfgr_csv", required=True,
                        help="TFGR feedback CSV (Phi_fb vs z)")
    parser.add_argument("--sn_csv", required=True,
                        help="Pantheon-like SN CSV with z,mu,mu_err")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    # 1. Load data
    z_tfgr, phi = load_tfgr_feedback(args.tfgr_csv)
    df_SN = pd.read_csv(args.sn_csv)
    z_SN = df_SN["z"].to_numpy()
    mu_SN = df_SN["mu"].to_numpy()
    mu_err = df_SN["mu_err"].to_numpy()

    # 2. Compute tau(z)
    z_fine, tau_fine = compute_tau(z_tfgr, phi)

    # 3. Fit k
    k_best, chi2_min, k_scan, chi2_scan, spline_tau = fit_k(
        z_fine, tau_fine, z_SN, mu_SN, mu_err
    )

    dof = len(z_SN) - 1
    chi2_red = chi2_min / dof

    # 4. Save summary
    df_sum = pd.DataFrame({
        "k_best":[k_best],
        "chi2_min":[chi2_min],
        "dof":[dof],
        "chi2_red":[chi2_red]
    })
    df_sum.to_csv(args.out_prefix+"_summary.csv", index=False)

    # 5. Plot chi2 vs k
    plt.figure(figsize=(8,5))
    plt.plot(k_scan, chi2_scan, '-')
    plt.axvline(k_best, color='r', ls='--',
                label=f"k_best={k_best:.2f}")
    plt.xlabel("k")
    plt.ylabel("chi^2")
    plt.title("Phase 176: chi^2 vs k (tau model)")
    plt.legend()
    plt.grid()
    plt.savefig(args.out_prefix+"_chi2_vs_k.png")
    plt.close()

    # 6. Plot tau(z)
    plt.figure(figsize=(8,5))
    plt.plot(z_fine, tau_fine)
    plt.xlabel("z")
    plt.ylabel("tau(z)")
    plt.title("Phase 176: tau(z) from TFGR feedback")
    plt.grid()
    plt.savefig(args.out_prefix+"_tau_z.png")
    plt.close()

    # 7. Compute best-fit Δμ(z)
    delta_mu_model = -k_best * spline_tau(z_SN)

    df_res = pd.DataFrame({
        "z": z_SN,
        "mu_SN": mu_SN,
        "mu_err": mu_err,
        "delta_mu_model": delta_mu_model
    })
    df_res.to_csv(args.out_prefix+"_delta_mu_fit.csv",
                  index=False)

    plt.figure(figsize=(10,6))
    plt.scatter(z_SN, mu_SN - np.mean(mu_SN), s=8, alpha=0.3,
                label="SN residuals (mean-subtracted)")
    plt.plot(z_SN, delta_mu_model - np.mean(delta_mu_model),
             'r-', label="TFGR τ-model Δμ(z)")
    plt.axhline(0, color='k', ls='--')
    plt.xlabel("z")
    plt.ylabel("Δμ(z)")
    plt.title("Phase 176: SN Δμ(z) vs TFGR τ-model")
    plt.legend()
    plt.grid()
    plt.savefig(args.out_prefix+"_delta_mu_fit_plot.png")
    plt.close()

    print("=== Phase 176 Completed ===")
    print(f"k_best = {k_best:.4f}")
    print(f"chi2_red = {chi2_red:.4f}")
    print(f"Saved outputs with prefix: {args.out_prefix}")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
