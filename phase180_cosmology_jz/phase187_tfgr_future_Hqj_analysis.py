#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 187: TFGR power-law component H(z), q(z), j(z) diagnostics
including future (z < 0) and comparison to ΛCDM.

ρ_TF(z) = Ω_TF0 (1+z)^n_TF,  n_TF = 3(1 + w_TF)

We assume a flat universe: Ω_r0 + Ω_m0 + Ω_TF0 = 1 (for TFGR),
and ΛCDM reference: Ω_r0 + Ω_m0 + Ω_Λ0 = 1.

Outputs:
  - CSV with z, H_TFGR, q_TFGR, j_TFGR, H_LCDM, q_LCDM, j_LCDM, Ω_i(z)
  - Plots: H(z), q(z), j(z) for TFGR vs ΛCDM

Author: ChatGPT (Phase 187)
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt


def compute_background_powerlaw(z, H0, Omega_r0, Omega_m0, Omega_TF0, n_TF):
    """
    Compute H(z), q(z), j(z) for TFGR power-law component.

    Components:
      - radiation: w_r = 1/3,  ρ_r ∝ (1+z)^4
      - matter   : w_m = 0,    ρ_m ∝ (1+z)^3
      - TFGR     : w_TF = n_TF/3 - 1, ρ_TF ∝ (1+z)^n_TF

    Returns:
      H_TF(z), q_TF(z), j_TF(z),
      Omega_r(z), Omega_m(z), Omega_TF(z)
    """
    z = np.asarray(z)
    ap1 = 1.0 + z  # a^{-1} = 1+z

    # equation of state parameters
    w_r = 1.0 / 3.0
    w_m = 0.0
    w_TF = n_TF / 3.0 - 1.0

    # density scalings
    rho_r = Omega_r0 * ap1**4
    rho_m = Omega_m0 * ap1**3
    rho_TF = Omega_TF0 * ap1**n_TF

    E2 = rho_r + rho_m + rho_TF  # = H^2 / H0^2 (flat)
    H = H0 * np.sqrt(E2)

    # fractional Ω_i(z)
    Om_r_z = rho_r / E2
    Om_m_z = rho_m / E2
    Om_TF_z = rho_TF / E2

    # deceleration parameter:
    # q = 1/2 Σ Ω_i(z) (1 + 3 w_i)
    q = 0.5 * (
        Om_r_z * (1.0 + 3.0 * w_r)
        + Om_m_z * (1.0 + 3.0 * w_m)
        + Om_TF_z * (1.0 + 3.0 * w_TF)
    )

    # jerk parameter for constant-w components:
    # j = 1 + (9/2) Σ Ω_i(z) w_i (1 + w_i)
    j = 1.0 + 4.5 * (
        Om_r_z * w_r * (1.0 + w_r)
        + Om_m_z * w_m * (1.0 + w_m)
        + Om_TF_z * w_TF * (1.0 + w_TF)
    )

    return H, q, j, Om_r_z, Om_m_z, Om_TF_z, w_TF


def compute_lcdm_reference(z, H0, Omega_r0, Omega_m0):
    """
    ΛCDM reference with w_Λ = -1 → ρ_Λ = const.

    Components:
      - radiation: w_r = 1/3
      - matter   : w_m = 0
      - Λ        : w_Λ = -1

    For ΛCDM with constant Λ, j(z) ≡ 1.

    Returns:
      H_LCDM(z), q_LCDM(z), j_LCDM(z), Ω_r(z), Ω_m(z), Ω_Λ(z)
    """
    z = np.asarray(z)
    ap1 = 1.0 + z

    w_r = 1.0 / 3.0
    w_m = 0.0
    w_L = -1.0

    Omega_L0 = 1.0 - Omega_r0 - Omega_m0

    rho_r = Omega_r0 * ap1**4
    rho_m = Omega_m0 * ap1**3
    rho_L = Omega_L0 * np.ones_like(z)

    E2 = rho_r + rho_m + rho_L
    H = H0 * np.sqrt(E2)

    Om_r_z = rho_r / E2
    Om_m_z = rho_m / E2
    Om_L_z = rho_L / E2

    q = 0.5 * (
        Om_r_z * (1.0 + 3.0 * w_r)
        + Om_m_z * (1.0 + 3.0 * w_m)
        + Om_L_z * (1.0 + 3.0 * w_L)
    )

    # ΛCDM with constant Λ: j ≡ 1
    j = np.ones_like(z)

    return H, q, j, Om_r_z, Om_m_z, Om_L_z, Omega_L0


def find_z_acc(z, q):
    """Find redshift where q(z) crosses 0 (acceleration onset)."""
    z = np.asarray(z)
    q = np.asarray(q)
    # index where |q| is minimal
    idx = np.argmin(np.abs(q))
    return z[idx], q[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 187: TFGR power-law H(z), q(z), j(z) diagnostics including future z"
    )
    parser.add_argument("--H0", type=float, required=True,
                        help="Hubble constant [km/s/Mpc]")
    parser.add_argument("--Omega_m0", type=float, required=True,
                        help="Present-day matter density parameter")
    parser.add_argument("--Omega_r0", type=float, required=True,
                        help="Present-day radiation density parameter")
    parser.add_argument("--Omega_TF0", type=float, required=True,
                        help="Present-day TFGR density parameter")
    parser.add_argument("--n_TF", type=float, required=True,
                        help="Power-law index for TFGR: ρ_TF ∝ (1+z)^n_TF")
    parser.add_argument("--z_min", type=float, required=True,
                        help="Minimum redshift (can be negative but > -1)")
    parser.add_argument("--z_max", type=float, required=True,
                        help="Maximum redshift")
    parser.add_argument("--n_z", type=int, default=400,
                        help="Number of z-grid points")
    parser.add_argument("--out_prefix", type=str, default="phase187_tfgr_future",
                        help="Prefix for output files")

    args = parser.parse_args()

    if args.z_min <= -1.0:
        raise ValueError("z_min must be greater than -1 (so that 1+z > 0).")

    # z-grid including future and past
    z_arr = np.linspace(args.z_min, args.z_max, args.n_z)

    print("===========================================")
    print("Phase 187: TFGR future H(z), q(z), j(z) diagnostics")
    print("-------------------------------------------")
    print(f"H0        = {args.H0:.3f} km/s/Mpc")
    print(f"Omega_m0  = {args.Omega_m0:.4f}")
    print(f"Omega_r0  = {args.Omega_r0:.4e}")
    print(f"Omega_TF0 = {args.Omega_TF0:.4f}")
    print(f"n_TF      = {args.n_TF:.4f}")
    print(f"z-range   = [{args.z_min:.2f}, {args.z_max:.2f}] (N={args.n_z})")
    print(f"out_prefix= {args.out_prefix}")
    print("===========================================")

    # --- TFGR power-law model ---
    H_TF, q_TF, j_TF, Om_r_TF, Om_m_TF, Om_TF, w_TF = compute_background_powerlaw(
        z_arr, args.H0, args.Omega_r0, args.Omega_m0, args.Omega_TF0, args.n_TF
    )

    # --- ΛCDM reference (same H0, Ω_m0, Ω_r0) ---
    H_LCDM, q_LCDM, j_LCDM, Om_r_LCDM, Om_m_LCDM, Om_L_LCDM, Omega_L0 = compute_lcdm_reference(
        z_arr, args.H0, args.Omega_r0, args.Omega_m0
    )

    # acceleration redshift (q=0) for TFGR and ΛCDM
    z_acc_TF, q_at_acc_TF = find_z_acc(z_arr, q_TF)
    z_acc_LCDM, q_at_acc_LCDM = find_z_acc(z_arr, q_LCDM)

    print(f"w_TF (from n_TF) = {w_TF:.4f}")
    print(f"ΛCDM ΩΛ0 (ref)   = {Omega_L0:.4f}")
    print("-------------------------------------------")
    print(f"TFGR  acceleration onset: z_acc ≈ {z_acc_TF:.3f}, q(z_acc) ≈ {q_at_acc_TF:.3e}")
    print(f"ΛCDM acceleration onset: z_acc ≈ {z_acc_LCDM:.3f}, q(z_acc) ≈ {q_at_acc_LCDM:.3e}")
    print("-------------------------------------------")

    # jerk diagnostics at a few key redshifts
    for z0 in [0.0, 0.5, 1.0, -0.5]:
        if (z0 < args.z_min) or (z0 > args.z_max):
            continue
        j_TF_val = np.interp(z0, z_arr, j_TF)
        print(f"j_TF(z={z0:+.2f}) ≈ {j_TF_val:.5f}")

    # --- Save CSV ---
    out_csv = f"{args.out_prefix}_Hqj_profile.csv"
    data = np.column_stack([
        z_arr,
        H_TF, q_TF, j_TF,
        H_LCDM, q_LCDM, j_LCDM,
        Om_r_TF, Om_m_TF, Om_TF
    ])
    header = (
        "z,"
        "H_TFGR,q_TFGR,j_TFGR,"
        "H_LCDM,q_LCDM,j_LCDM,"
        "Omega_r_TFGR,Omega_m_TFGR,Omega_TF_TFGR"
    )
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"[INFO] Saved diagnostic CSV to {out_csv}")

    # --- Plot H(z) ---
    plt.figure(figsize=(8, 6))
    plt.plot(z_arr, H_TF, label="TFGR power-law", lw=2)
    plt.plot(z_arr, H_LCDM, "--", label="ΛCDM", lw=2)
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("H(z) diagnostics: TFGR vs ΛCDM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
    plt.tight_layout()
    out_H = f"{args.out_prefix}_H_z.png"
    plt.savefig(out_H, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_H}")

    # --- Plot q(z) ---
    plt.figure(figsize=(8, 6))
    plt.plot(z_arr, q_TF, label="TFGR power-law", lw=2)
    plt.plot(z_arr, q_LCDM, "--", label="ΛCDM", lw=2)
    plt.axhline(0.0, color="k", linestyle=":", linewidth=1)
    plt.axvline(z_acc_TF, color="C0", linestyle=":", linewidth=1,
                label=f"TFGR z_acc≈{z_acc_TF:.2f}")
    plt.axvline(z_acc_LCDM, color="C1", linestyle=":", linewidth=1,
                label=f"ΛCDM z_acc≈{z_acc_LCDM:.2f}")
    plt.xlabel("z")
    plt.ylabel("q(z)")
    plt.title("Deceleration parameter q(z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_q = f"{args.out_prefix}_q_z.png"
    plt.savefig(out_q, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_q}")

    # --- Plot j(z) ---
    plt.figure(figsize=(8, 6))
    plt.plot(z_arr, j_TF, label="TFGR power-law", lw=2)
    plt.axhline(1.0, color="C1", linestyle="--", linewidth=2,
                label="ΛCDM j(z)=1")
    plt.xlabel("z")
    plt.ylabel("j(z)")
    plt.title("Jerk parameter j(z)")
    plt.grid(True, alpha=0.3)
    plt.axvline(0.0, color="k", linestyle=":", linewidth=1)
    plt.tight_layout()
    out_j = f"{args.out_prefix}_j_z.png"
    plt.savefig(out_j, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_j}")

    print("Done.")


if __name__ == "__main__":
    main()
