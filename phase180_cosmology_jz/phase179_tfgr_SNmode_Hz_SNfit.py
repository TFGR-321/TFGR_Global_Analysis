#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 179: SN-mode based extra DE component + SN fit

Use the TFGR SN-mode energy profile rho_SN(z) as an additional
dark-energy-like component in a flat FRW model and fit Pantheon SNe.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

C_LIGHT_KM_S = 299792.458  # speed of light [km/s]


def load_sn_csv(path):
    """
    Expect a Pantheon-like CSV with at least columns:
    z, mu, mu_err (or mB,mB_err etc. but here assume mu, mu_err).
    """
    df = pd.read_csv(path)
    # Try to infer column names
    if {"z", "mu", "mu_err"}.issubset(df.columns):
        z = df["z"].values
        mu = df["mu"].values
        mu_err = df["mu_err"].values
    else:
        # Fallback for typical Pantheon naming: zHD, mB, dmb
        z_col = "z"
        if "zHD" in df.columns:
            z_col = "zHD"
        mu_col = "mu"
        if "mB" in df.columns and "dmb" in df.columns:
            # If only apparent mags are present, user should have precomputed mu;
            # here we just use mB as "mu" for differential fitting.
            mu_col = "mB"
        err_col = "mu_err"
        if "dmb" in df.columns:
            err_col = "dmb"

        z = df[z_col].values
        mu = df[mu_col].values
        mu_err = df[err_col].values

    return z, mu, mu_err


def load_snmode_profile(path):
    """
    Load TFGR SN-mode energy profile CSV.
    Expect columns:
      z, rho_SN
    """
    df = pd.read_csv(path)
    # Try to guess column names
    z_col = "z"
    if "redshift" in df.columns:
        z_col = "redshift"

    if "rho_SN" in df.columns:
        rho_col = "rho_SN"
    else:
        # try something like rho or rho_sn
        cand = [c for c in df.columns if "rho" in c.lower()]
        if not cand:
            raise RuntimeError("Could not find rho_SN column in SN-mode CSV")
        rho_col = cand[0]

    z = df[z_col].values
    rho = df[rho_col].values

    # Sort by z just in case
    idx = np.argsort(z)
    return z[idx], rho[idx]


def build_H_function(H0, Om0, Or0, OmSN0, z_snmode, rho_sn):
    """
    Build a function H(z) using FRW + extra SN-mode component.

    H^2/H0^2 = Or0(1+z)^4 + Om0(1+z)^3 + OmLambda0 + OmSN0 * f_SN(z),
    with OmLambda0 chosen so that flatness holds.
    f_SN(z) = rho_SN(z)/rho_SN(0).
    """
    # Normalised SN-mode shape
    rho0 = rho_sn[0]
    if rho0 == 0.0:
        raise RuntimeError("rho_SN(0) is zero; cannot normalise SN-mode profile")
    f_sn_raw = rho_sn / rho0

    # Interpolator for f_SN(z)
    z_min = np.min(z_snmode)
    z_max = np.max(z_snmode)

    f_sn_interp = interp1d(
        z_snmode,
        f_sn_raw,
        kind="linear",
        bounds_error=False,
        fill_value=(f_sn_raw[0], f_sn_raw[-1]),
    )

    # Flatness condition
    OmSN0 = float(OmSN0)
    OmLambda0 = 1.0 - Om0 - Or0 - OmSN0

    def H_of_z(z):
        z = np.asarray(z)
        f_sn = f_sn_interp(z)
        Omeff = Or0 * (1.0 + z) ** 4 + Om0 * (1.0 + z) ** 3 + OmLambda0 + OmSN0 * f_sn
        return H0 * np.sqrt(Omeff)

    return H_of_z, OmLambda0


def luminosity_distance_FRW(z, H_of_z, z_max_int=None):
    """
    Compute luminosity distance d_L(z) for flat FRW with given H(z).
    Use line-of-sight comoving distance integral via scipy.integrate.quad.

    d_L(z) = (1+z) * c/H0 * integral_0^z dz'/E(z')
    where E(z) = H(z)/H0.

    We do not explicitly need H0 inside this function; H(z) already has units
    km/s/Mpc, so we can use:

    d_L(z) [Mpc] = (1+z) * c * integral_0^z dz' / H(z').

    """
    # We do scalar integration for each z.
    # z_max_int is optional safety for integration range.
    def dL_single(zz):
        def integrand(zp):
            return C_LIGHT_KM_S / H_of_z(zp)

        upper = float(zz)
        if z_max_int is not None:
            upper = min(upper, z_max_int)

        integral, _ = quad(integrand, 0.0, upper, limit=200)
        return (1.0 + zz) * integral

    # Support array input
    z = np.atleast_1d(z)
    dL = np.array([dL_single(zz) for zz in z])
    return dL  # in Mpc, because c/H ~ Mpc


def mu_from_dL(dL_Mpc):
    """Distance modulus mu = 5 log10(d_L / 10 pc)."""
    return 5.0 * (np.log10(dL_Mpc) + 5.0)


def chi2_with_best_C(mu_sn, mu_err, mu_model):
    """
    Compute chi^2 minimised over additive constant C:
    mu_sn ~ mu_model + C.
    """
    w = 1.0 / (mu_err ** 2)
    # Best C (weighted)
    C_best = np.sum(w * (mu_sn - mu_model)) / np.sum(w)
    resid = mu_sn - (mu_model + C_best)
    chi2 = np.sum((resid / mu_err) ** 2)
    return chi2, C_best, resid


def main():
    parser = argparse.ArgumentParser(
        description="Phase 179: TFGR SN-mode extra DE + SN fit"
    )
    parser.add_argument("--sn_csv", required=True, help="Pantheon SN CSV")
    parser.add_argument(
        "--snmode_csv",
        required=True,
        help="TFGR SN-mode rho_SN(z) profile CSV (z, rho_SN)",
    )
    parser.add_argument("--H0", type=float, default=70.0, help="H0 [km/s/Mpc]")
    parser.add_argument("--Omega_m0", type=float, default=0.3, help="Omega_m0 today")
    parser.add_argument(
        "--Omega_r0",
        type=float,
        default=1.0e-4,
        help="Omega_r0 today (radiation)",
    )
    parser.add_argument(
        "--OmSN_min",
        type=float,
        default=0.0,
        help="min Omega_SN0 for scan",
    )
    parser.add_argument(
        "--OmSN_max",
        type=float,
        default=0.2,
        help="max Omega_SN0 for scan",
    )
    parser.add_argument(
        "--OmSN_steps",
        type=int,
        default=41,
        help="number of steps in Omega_SN0 scan",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=0.01,
        help="minimal z for SN to use",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=1.5,
        help="maximal z for SN to use",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="prefix for output files",
    )

    args = parser.parse_args()

    print("=== Phase 179: TFGR SN-mode extra DE + SN fit ===")
    print(f"SN CSV      : {args.sn_csv}")
    print(f"SN-mode CSV : {args.snmode_csv}")
    print(f"H0          : {args.H0:.3f} km/s/Mpc")
    print(f"Omega_m0    : {args.Omega_m0:.5f}")
    print(f"Omega_r0    : {args.Omega_r0:.5e}")
    print(
        f"Omega_SN0 scan: [{args.OmSN_min}, {args.OmSN_max}] "
        f"({args.OmSN_steps} steps)"
    )
    print(
        f"SN z-range used: [{args.z_min}, {args.z_max}] "
    )
    print("===============================================")

    # 1) Load data
    z_sn, mu_sn, mu_err = load_sn_csv(args.sn_csv)
    z_mode, rho_mode = load_snmode_profile(args.snmode_csv)

    # Cut SN to desired z-range and to range covered by SN-mode
    z_min_mode = np.min(z_mode)
    z_max_mode = np.max(z_mode)

    z_low = max(args.z_min, z_min_mode)
    z_high = min(args.z_max, z_max_mode)

    mask = (z_sn >= z_low) & (z_sn <= z_high)
    z_sn_used = z_sn[mask]
    mu_sn_used = mu_sn[mask]
    mu_err_used = mu_err[mask]

    print(f"N_SN total  = {len(z_sn)}")
    print(
        f"N_SN used   = {len(z_sn_used)} in z ∈ "
        f"[{z_low:.3f}, {z_high:.3f}] and mode z-range"
    )

    if len(z_sn_used) == 0:
        raise RuntimeError("No SN left after z-range cut. Check settings.")

    # 2) Scan Omega_SN0
    Om0 = args.Omega_m0
    Or0 = args.Omega_r0
    H0 = args.H0

    OmSN_grid = np.linspace(args.OmSN_min, args.OmSN_max, args.OmSN_steps)

    chi2_list = []
    C_best_list = []
    OmLambda_list = []

    best_idx = None
    chi2_best = None

    # Precompute for interpolation of d_L(z) on a fine grid to speed up
    z_int_max = z_high * 1.05

    for i, OmSN0 in enumerate(OmSN_grid):
        H_of_z, OmLambda0 = build_H_function(
            H0, Om0, Or0, OmSN0, z_mode, rho_mode
        )
        OmLambda_list.append(OmLambda0)

        # Compute d_L for all SN points (can be optimised, but OK for now)
        dL = luminosity_distance_FRW(z_sn_used, H_of_z, z_max_int=z_int_max)
        mu_model = mu_from_dL(dL)

        chi2, C_best, resid = chi2_with_best_C(
            mu_sn_used, mu_err_used, mu_model
        )
        chi2_list.append(chi2)
        C_best_list.append(C_best)

        if (chi2_best is None) or (chi2 < chi2_best):
            chi2_best = chi2
            best_idx = i

        print(
            f"[{i+1:3d}/{len(OmSN_grid):3d}] "
            f"Omega_SN0={OmSN0: .5f}, Omega_Lambda0={OmLambda0: .5f}, "
            f"chi2={chi2: .3f}"
        )

    chi2_array = np.array(chi2_list)
    C_best_array = np.array(C_best_list)
    OmLambda_array = np.array(OmLambda_list)

    OmSN_best = OmSN_grid[best_idx]
    OmLambda_best = OmLambda_array[best_idx]
    C_best = C_best_array[best_idx]

    dof = len(z_sn_used) - 1  # only C is effectively fitted here
    chi2_red_best = chi2_best / dof

    print("------ Best-fit results ------")
    print(f"Omega_SN0_best   = {OmSN_best:.6f}")
    print(f"Omega_Lambda0_best = {OmLambda_best:.6f}")
    print(f"C_best           = {C_best:.6f} mag")
    print(f"chi2_min         = {chi2_best:.3f}")
    print(f"dof              = {dof}")
    print(f"chi2_red         = {chi2_red_best:.5f}")
    print("------------------------------")

    # 3) Save summary CSV
    summary = pd.DataFrame(
        {
            "Omega_SN0": OmSN_grid,
            "Omega_Lambda0": OmLambda_array,
            "chi2": chi2_array,
            "C_best": C_best_array,
        }
    )
    summary_out = f"{args.out_prefix}_scan_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"[Phase 179] Saved scan summary CSV -> {summary_out}")

    # 4) Compute and save residuals for best model
    H_best, _ = build_H_function(H0, Om0, Or0, OmSN_best, z_mode, rho_mode)
    dL_best = luminosity_distance_FRW(z_sn_used, H_best, z_max_int=z_int_max)
    mu_model_best = mu_from_dL(dL_best) + C_best
    resid_best = mu_sn_used - mu_model_best

    resid_df = pd.DataFrame(
        {
            "z": z_sn_used,
            "mu_SN": mu_sn_used,
            "mu_err": mu_err_used,
            "mu_model_best": mu_model_best,
            "residual": resid_best,
        }
    )
    resid_out = f"{args.out_prefix}_best_residuals.csv"
    resid_df.to_csv(resid_out, index=False)
    print(f"[Phase 179] Saved best-fit residuals CSV -> {resid_out}")

    # 5) Plot chi^2 vs Omega_SN0
    plt.figure(figsize=(7, 5))
    plt.plot(OmSN_grid, chi2_array, marker="o", linestyle="-")
    plt.axvline(OmSN_best, color="r", linestyle="--", label=f"best = {OmSN_best:.3f}")
    plt.xlabel(r"$\Omega_{\mathrm{SN},0}$")
    plt.ylabel(r"$\chi^2$")
    plt.title("Phase 179: chi^2 vs Omega_SN0 (SN-mode extra DE)")
    plt.legend()
    plt.tight_layout()
    chi2_plot = f"{args.out_prefix}_chi2_vs_OmSN0.png"
    plt.savefig(chi2_plot, dpi=150)
    print(f"[Phase 179] Saved chi^2 plot -> {chi2_plot}")
    plt.close()

    # 6) Plot residuals vs z for best model
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        z_sn_used,
        resid_best,
        yerr=mu_err_used,
        fmt=".",
        alpha=0.6,
        label="SN - model",
    )
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xlabel("z")
    plt.ylabel(r"$\mu_{\rm SN} - \mu_{\rm model}$ [mag]")
    plt.title("Phase 179: SN residuals for best SN-mode extra DE model")
    plt.legend()
    plt.tight_layout()
    resid_plot = f"{args.out_prefix}_best_residuals.png"
    plt.savefig(resid_plot, dpi=150)
    print(f"[Phase 179] Saved residuals plot -> {resid_plot}")
    plt.close()


if __name__ == "__main__":
    main()
