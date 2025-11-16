#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 172: Reconstruct extra time-field component from SN residuals

Inputs
------
- tfgr_csv : phase161_tfgr_plateau_H0_70_Hz_qz_Omegaz_mu.csv
             (columns: z, H_z_km_s_Mpc, q_z, Omega_m_z, Omega_r_z,
                       Omega_t_z, D_C_Mpc, D_L_Mpc, mu)
- sn_csv   : pantheon_SN.csv
             (columns: z, mu, mu_err)

Outputs (prefix = --out_prefix)
-------
- prefix + "_delta_mu_fit.csv"
    z_grid, delta_mu_fit, delta_D_over_D, H_eff, H_tfgr, Omega_extra
- prefix + "_delta_mu_data.csv"
    z_sn, delta_mu_data, mu_sn, mu_tfgr_interp, mu_err, weight

- prefix + "_delta_mu_fit.png"
    SN residuals Δμ(z) + best-fit curve
- prefix + "_delta_D_over_D.png"
    fractional distance correction δD/D vs z
- prefix + "_H_comparison.png"
    H_eff(z) vs H_TFGR(z)
- prefix + "_Omega_extra.png"
    Ω_extra(z) vs z
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


C_LIGHT = 299792.458  # km/s


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 172: reconstruct extra time-field from SN residuals"
    )
    p.add_argument("--tfgr_csv", required=True,
                   help="TFGR background CSV (from phase161)")
    p.add_argument("--sn_csv", required=True,
                   help="Pantheon SN CSV (z, mu, mu_err)")
    p.add_argument("--H0", type=float, default=70.0,
                   help="H0 [km/s/Mpc] (for dimensionless Omega_extra)")
    p.add_argument("--z_min", type=float, default=0.0,
                   help="minimum z to use")
    p.add_argument("--z_max", type=float, default=1.5,
                   help="maximum z to use (must be <= max z of TFGR grid)")
    p.add_argument("--out_prefix", required=True,
                   help="output file prefix")
    return p.parse_args()


def load_tfgr(tfgr_csv):
    df = pd.read_csv(tfgr_csv)
    required = ["z", "H_z_km_s_Mpc", "D_L_Mpc", "mu"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"TFGR CSV missing column: {col}")
    return df


def load_sn(sn_csv):
    df = pd.read_csv(sn_csv)
    required = ["z", "mu", "mu_err"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"SN CSV missing column: {col}")
    return df


def weighted_offset_fit(mu_sn, mu_model, sigma):
    """return best-fit constant offset C minimizing sum((mu_sn - (mu_model+C))^2/sigma^2)."""
    w = 1.0 / (sigma**2)
    # minimize chi2(C) = sum w (mu_sn - mu_model - C)^2
    num = np.sum(w * (mu_sn - mu_model))
    den = np.sum(w)
    C_best = num / den
    return C_best


def delta_mu_model(z, A, B):
    """Parametric model for Δμ(z):  A * (z/(1+z))^B."""
    x = z / (1.0 + z)
    return A * (x ** B)


def reconstruct_phase172(args):
    # 1. load data
    tfgr = load_tfgr(args.tfgr_csv)
    sn = load_sn(args.sn_csv)

    # restrict TFGR z-range
    tfgr = tfgr[(tfgr["z"] >= args.z_min) & (tfgr["z"] <= args.z_max)].copy()
    tfgr = tfgr.sort_values("z").reset_index(drop=True)

    # restrict SN to same range
    sn = sn[(sn["z"] >= args.z_min) & (sn["z"] <= args.z_max)].copy()
    sn = sn.sort_values("z").reset_index(drop=True)

    if len(sn) == 0:
        raise RuntimeError("No SN left after z-range cut.")

    print(f"[Phase 172] N_SN used = {len(sn)} "
          f"in z ∈ [{sn['z'].min():.3f}, {sn['z'].max():.3f}]")

    # 2. interpolate TFGR mu(z) and D_L(z) onto SN z
    z_tf = tfgr["z"].to_numpy()
    mu_tf = tfgr["mu"].to_numpy()
    D_L_tf = tfgr["D_L_Mpc"].to_numpy()
    H_tf = tfgr["H_z_km_s_Mpc"].to_numpy()

    z_sn = sn["z"].to_numpy()
    mu_sn = sn["mu"].to_numpy()
    mu_err = sn["mu_err"].to_numpy()

    mu_tf_interp = np.interp(z_sn, z_tf, mu_tf)
    D_L_tf_interp = np.interp(z_sn, z_tf, D_L_tf)

    # 3. best-fit constant offset C between SN and TFGR model
    C_best = weighted_offset_fit(mu_sn, mu_tf_interp, mu_err)
    print(f"[Phase 172] C_best (SN - TFGR) = {C_best:.4f} mag")

    # 4. Δμ_i = μ_SN - ( μ_TFGR + C_best )
    delta_mu_data = mu_sn - (mu_tf_interp + C_best)

    # 5. Fit Δμ(z) = A (z/(1+z))^B
    #    初期値は SN 残差のスケールから適当に。
    A_init = np.median(delta_mu_data)
    if not np.isfinite(A_init) or abs(A_init) < 1e-3:
        A_init = -0.2
    B_init = 1.0

    print(f"[Phase 172] Initial guess for (A,B) = ({A_init:.3f}, {B_init:.3f})")

    # 安全のため、エラーが変でもとりあえず 0.1mag の最低値を与える
    sigma_fit = np.maximum(mu_err, 0.1)

    try:
        popt, pcov = curve_fit(
            delta_mu_model,
            z_sn,
            delta_mu_data,
            p0=[A_init, B_init],
            sigma=sigma_fit,
            absolute_sigma=True,
            maxfev=10000,
        )
        A_best, B_best = popt
        perr = np.sqrt(np.diag(pcov))
        dA, dB = perr
    except Exception as e:
        print("[Phase 172] WARNING: curve_fit failed:", e)
        A_best, B_best = A_init, B_init
        dA = dB = np.nan

    print(f"[Phase 172] Best-fit Δμ(z) params:")
    print(f"  A = {A_best:.4f} ± {dA:.4f}")
    print(f"  B = {B_best:.4f} ± {dB:.4f}")

    # 6. make fine z-grid for model comparison
    z_grid = np.linspace(tfgr["z"].min(), tfgr["z"].max(), 400)
    delta_mu_fit = delta_mu_model(z_grid, A_best, B_best)

    # 7. convert Δμ(z) -> δD/D(z)
    delta_D_over_D_grid = 10.0 ** (delta_mu_fit / 5.0) - 1.0

    # 8. reconstruct effective D_L(z) and H_eff(z)
    D_L_tf_grid = np.interp(z_grid, z_tf, D_L_tf)
    H_tf_grid = np.interp(z_grid, z_tf, H_tf)

    D_L_eff_grid = D_L_tf_grid * (1.0 + delta_D_over_D_grid)
    # comoving distance
    D_C_eff = D_L_eff_grid / (1.0 + z_grid)

    # numerical derivative dD_C/dz
    dDdz = np.gradient(D_C_eff, z_grid)
    H_eff = C_LIGHT / dDdz  # km/s/Mpc

    # 9. extra component Ω_extra(z)
    H0 = args.H0
    Omega_extra = (H_eff**2 - H_tf_grid**2) / (H0**2)

    # 10. save CSVs
    out_delta = args.out_prefix + "_delta_mu_fit.csv"
    df_delta = pd.DataFrame({
        "z": z_grid,
        "delta_mu_fit": delta_mu_fit,
        "delta_D_over_D": delta_D_over_D_grid,
        "D_L_TFGR_Mpc": D_L_tf_grid,
        "D_L_eff_Mpc": D_L_eff_grid,
        "H_TFGR_km_s_Mpc": H_tf_grid,
        "H_eff_km_s_Mpc": H_eff,
        "Omega_extra": Omega_extra,
    })
    df_delta.to_csv(out_delta, index=False)
    print(f"[Phase 172] Saved model grid -> {out_delta}")

    out_data = args.out_prefix + "_delta_mu_data.csv"
    df_data = pd.DataFrame({
        "z_sn": z_sn,
        "mu_sn": mu_sn,
        "mu_err": mu_err,
        "mu_tfgr_interp": mu_tf_interp,
        "C_best": np.full_like(z_sn, C_best),
        "delta_mu_data": delta_mu_data,
        "weight": 1.0 / sigma_fit**2,
    })
    df_data.to_csv(out_data, index=False)
    print(f"[Phase 172] Saved SN residuals -> {out_data}")

    # 11. plots
    # (a) Δμ(z) data + fit
    plt.figure(figsize=(8, 5))
    plt.errorbar(z_sn, delta_mu_data, yerr=mu_err,
                 fmt="o", ms=2, alpha=0.5, label="SN data")
    plt.plot(z_grid, delta_mu_fit, "r-", lw=2, label="best-fit Δμ(z)")
    plt.axhline(0.0, color="k", ls="--", lw=1)
    plt.xlabel("z")
    plt.ylabel(r"$\Delta\mu(z) = \mu_{\rm SN} - \mu_{\rm TFGR} - C_{\rm best}$")
    plt.title("Phase 172: SN residuals and best-fit Δμ(z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_delta_mu_fit.png", dpi=150)
    plt.close()

    # (b) δD/D(z)
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, delta_D_over_D_grid, lw=2)
    plt.axhline(0.0, color="k", ls="--", lw=1)
    plt.xlabel("z")
    plt.ylabel(r"$\delta_D(z) = D_{\rm SN}/D_{\rm TFGR} - 1$")
    plt.title("Phase 172: fractional distance correction δD/D")
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_delta_D_over_D.png", dpi=150)
    plt.close()

    # (c) H_eff vs H_TFGR
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, H_tf_grid, "k--", lw=2, label="H_TFGR(z)")
    plt.plot(z_grid, H_eff, "r-", lw=2, label="H_eff(z) from SN+TFGR")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("Phase 172: H_eff(z) reconstructed vs TFGR H(z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_H_comparison.png", dpi=150)
    plt.close()

    # (d) Ω_extra(z)
    plt.figure(figsize=(8, 5))
    plt.plot(z_grid, Omega_extra, lw=2)
    plt.axhline(0.0, color="k", ls="--", lw=1)
    plt.xlabel("z")
    plt.ylabel(r"$\Omega_{\rm extra}(z)$")
    plt.title("Phase 172: effective extra time-field density Ω_extra(z)")
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_Omega_extra.png", dpi=150)
    plt.close()

    print("[Phase 172] All done.")


def main():
    args = parse_args()
    reconstruct_phase172(args)


if __name__ == "__main__":
    main()
