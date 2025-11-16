#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
phase185_tfgr_powerlaw_Hqj_diagnostics.py

Analytic TFGR power-law component:
    rho_TF(z) = Omega_TF0 * (1+z)^n

Compute H(z), q(z), j(z) and compare with ΛCDM (j=1).
Outputs:
  - CSV of z, H_TF, q_TF, j_TF, H_LCDM, q_LCDM
  - Three PNG figures:
      * ..._H_z.png
      * ..._q_z.png
      * ..._j_z.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

def compute_H_q_j(z, H0, Om0, Or0, Otf0, n_tf):
    """
    Return H(z), q(z), j(z) for TFGR power-law model.

    E2 = H^2 / H0^2
    E2(z) = Or0 (1+z)^4 + Om0 (1+z)^3 + Otf0 (1+z)^n_tf
    """
    zp1 = 1.0 + z

    Ar = Or0 * zp1**4
    Am = Om0 * zp1**3
    At = Otf0 * zp1**n_tf

    E2 = Ar + Am + At

    # S1 = sum n_i A_i
    S1 = 4.0 * Ar + 3.0 * Am + n_tf * At
    # S2 = sum n_i^2 A_i
    S2 = 16.0 * Ar + 9.0 * Am + (n_tf**2) * At

    # H(z)
    H = H0 * np.sqrt(E2)

    # q(z) = -1 + S1 / (2 E2)
    q = -1.0 + S1 / (2.0 * E2)

    # j(z) = (2E2 - 3S1 + S2) / (2E2)
    j = (2.0 * E2 - 3.0 * S1 + S2) / (2.0 * E2)

    return H, q, j


def compute_lcdm_H_q(z, H0, Om0, Or0):
    """
    Standard flat ΛCDM with
      Omega_m0 = Om0
      Omega_r0 = Or0
      Omega_L0 = 1 - Om0 - Or0
    """
    zp1 = 1.0 + z
    Ol0 = 1.0 - Om0 - Or0

    Ar = Or0 * zp1**4
    Am = Om0 * zp1**3
    Al = Ol0 * np.ones_like(z)

    E2 = Ar + Am + Al

    # S1 = 4 Ar + 3 Am + 0 * Al
    S1 = 4.0 * Ar + 3.0 * Am

    H = H0 * np.sqrt(E2)
    q = -1.0 + S1 / (2.0 * E2)

    # j(z) = 1 identically (not needed numerically)
    j = np.ones_like(z)

    return H, q, j


def main():
    parser = argparse.ArgumentParser(
        description="TFGR power-law H(z), q(z), j(z) diagnostics"
    )
    parser.add_argument("--H0", type=float, default=70.0,
                        help="Hubble constant today [km/s/Mpc]")
    parser.add_argument("--Omega_m0", type=float, default=0.3,
                        help="Matter density parameter today")
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4,
                        help="Radiation density parameter today")
    parser.add_argument("--Omega_TF0", type=float, default=0.7,
                        help="TFGR power-law component density today")
    parser.add_argument("--n_TF", type=float, default=0.0,
                        help="Exponent n in rho_TF ~ (1+z)^n")
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--n_z", type=int, default=400)
    parser.add_argument("--out_prefix", type=str,
                        default="phase185_tfgr_powerlaw")
    args = parser.parse_args()

    H0 = args.H0
    Om0 = args.Omega_m0
    Or0 = args.Omega_r0
    Otf0 = args.Omega_TF0
    n_tf = args.n_TF

    # consistency check
    if Om0 + Or0 + Otf0 > 1.0 + 1e-6:
        print("[WARN] Om0 + Or0 + OTF0 > 1: flatness violated.")

    # effective equation-of-state of TF component
    w_tf = n_tf / 3.0 - 1.0

    print("===================================")
    print("TFGR power-law model diagnostics")
    print(f"H0        = {H0:.3f} km/s/Mpc")
    print(f"Omega_m0  = {Om0:.4f}")
    print(f"Omega_r0  = {Or0:.4e}")
    print(f"Omega_TF0 = {Otf0:.4f}")
    print(f"n_TF      = {n_tf:.4f}")
    print(f"w_TF      = {w_tf:.4f}  (since n=3(1+w_TF))")
    print("===================================")

    # redshift grid
    z = np.linspace(args.z_min, args.z_max, args.n_z)

    # TFGR diagnostics
    H_tf, q_tf, j_tf = compute_H_q_j(z, H0, Om0, Or0, Otf0, n_tf)

    # ΛCDM reference
    H_lcdm, q_lcdm, j_lcdm = compute_lcdm_H_q(z, H0, Om0, Or0)

    # j0 (today)
    j0_tf = float(j_tf[0])
    print(f"j_TF(z=0)   = {j0_tf:.5f}")
    print("For ΛCDM, j(z) ≡ 1 at all z.")

    # ---- CSV output ----
    csv_name = f"{args.out_prefix}_Hqj_profile.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "z",
            "H_TF", "q_TF", "j_TF",
            "H_LCDM", "q_LCDM", "j_LCDM"
        ])
        for zi, htf, qt, jt, hl, ql, jl in zip(
                z, H_tf, q_tf, j_tf, H_lcdm, q_lcdm, j_lcdm):
            writer.writerow([zi, htf, qt, jt, hl, ql, jl])
    print(f"[INFO] Saved diagnostic profile to {csv_name}")

    # ---- Plot H(z) ----
    plt.figure()
    plt.plot(z, H_tf, label="TFGR power-law")
    plt.plot(z, H_lcdm, linestyle="--", label="ΛCDM")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("H(z) diagnostics: TFGR vs ΛCDM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_H_z.png", dpi=200)
    plt.close()

    # ---- Plot q(z) ----
    plt.figure()
    plt.plot(z, q_tf, label="TFGR power-law")
    plt.plot(z, q_lcdm, linestyle="--", label="ΛCDM")
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("z")
    plt.ylabel("q(z)")
    plt.title("Deceleration parameter q(z)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_q_z.png", dpi=200)
    plt.close()

    # ---- Plot j(z) ----
    plt.figure()
    plt.plot(z, j_tf, label="TFGR power-law")
    plt.axhline(1.0, linestyle="--", label="ΛCDM j(z)=1")
    plt.xlabel("z")
    plt.ylabel("j(z)")
    plt.title("Jerk parameter j(z)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_j_z.png", dpi=200)
    plt.close()

    print("[INFO] Saved plots:")
    print(f"  {args.out_prefix}_H_z.png")
    print(f"  {args.out_prefix}_q_z.png")
    print(f"  {args.out_prefix}_j_z.png")
    print("Done.")


if __name__ == "__main__":
    main()
