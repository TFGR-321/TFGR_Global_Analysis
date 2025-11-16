#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 183: TFGR cosmological diagnostics
- Compute H(z), q(z), j(z) (jerk) for a given TFGR history
- Optionally overlay ΛCDM comparison (H(z), q(z), j=1)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_dq_dz(z, q):
    """
    Numerically compute dq/dz using numpy.gradient.
    z : 1D array (monotonic)
    q : 1D array
    """
    dz = np.gradient(z)
    dq = np.gradient(q)
    dq_dz = dq / dz
    return dq_dz


def compute_jerk_from_q(z, q):
    """
    Use cosmography relation:
        j(z) = (1+z) * dq/dz + q + 2 q^2
    """
    dq_dz = compute_dq_dz(z, q)
    j = (1.0 + z) * dq_dz + q + 2.0 * q**2
    return j


def lcdm_H_q_j(z, H0, Om0, Or0):
    """
    Compute H_LCDM(z), q_LCDM(z), j_LCDM(z) for flat ΛCDM.
    - Ω_Λ0 = 1 - Ω_m0 - Ω_r0
    - H(z) = H0 * sqrt(Ω_r (1+z)^4 + Ω_m (1+z)^3 + Ω_Λ)
    - q(z) = 0.5 Ω_m(z) + Ω_r(z) - Ω_Λ(z)
      (for flat ΛCDM)
    - j(z) = 1  (constant for ΛCDM)
    """
    Ol0 = 1.0 - Om0 - Or0
    zp1 = 1.0 + z
    Ez2 = Or0 * zp1**4 + Om0 * zp1**3 + Ol0
    Hz = H0 * np.sqrt(Ez2)

    # time-dependent density parameters
    Om_z = Om0 * zp1**3 / Ez2
    Or_z = Or0 * zp1**4 / Ez2
    Ol_z = Ol0 / Ez2

    qz = 0.5 * Om_z + Or_z - Ol_z
    jz = np.ones_like(z)  # constant jerk for ΛCDM

    return Hz, qz, jz


def main():
    parser = argparse.ArgumentParser(
        description="Phase 183: TFGR diagnostics H(z), q(z), j(z)"
    )
    parser.add_argument(
        "--history_csv",
        type=str,
        required=True,
        help="TFGR history CSV (must contain columns z, H_z_km_s_Mpc, q_z)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Prefix for output files",
    )
    parser.add_argument(
        "--H0",
        type=float,
        default=None,
        help="H0 for ΛCDM comparison (km/s/Mpc). If omitted, no ΛCDM overlay.",
    )
    parser.add_argument(
        "--Omega_m0",
        type=float,
        default=None,
        help="Omega_m0 for ΛCDM comparison.",
    )
    parser.add_argument(
        "--Omega_r0",
        type=float,
        default=0.0,
        help="Omega_r0 for ΛCDM comparison (default 0.0).",
    )

    args = parser.parse_args()

    print("=== Phase 183: TFGR diagnostics (H, q, j) ===")
    print(f"history_csv = {args.history_csv}")
    print(f"out_prefix  = {args.out_prefix}")
    if args.H0 is not None and args.Omega_m0 is not None:
        print("ΛCDM comparison: ON")
        print(f"H0        = {args.H0:.3f} km/s/Mpc")
        print(f"Omega_m0  = {args.Omega_m0:.5f}")
        print(f"Omega_r0  = {args.Omega_r0:.5f}")
    else:
        print("ΛCDM comparison: OFF")
    print("============================================")

    # --- Load TFGR history ---
    df = pd.read_csv(args.history_csv)
    required_cols = ["z", "H_z_km_s_Mpc", "q_z"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Required column '{c}' not found in {args.history_csv}. "
                f"Available columns: {list(df.columns)}"
            )

    # sort by z just in case
    df = df.sort_values("z").reset_index(drop=True)

    z = df["z"].to_numpy()
    H_tfgr = df["H_z_km_s_Mpc"].to_numpy()
    q_tfgr = df["q_z"].to_numpy()

    # --- Compute jerk j(z) ---
    j_tfgr = compute_jerk_from_q(z, q_tfgr)

    # --- Save diagnostics CSV ---
    diag_df = pd.DataFrame(
        {
            "z": z,
            "H_z_TFGR": H_tfgr,
            "q_z_TFGR": q_tfgr,
            "j_z_TFGR": j_tfgr,
        }
    )
    out_csv = f"{args.out_prefix}_diagnostics.csv"
    diag_df.to_csv(out_csv, index=False)
    print(f"[Phase 183] Saved diagnostics CSV -> {out_csv}")

    # --- Optional ΛCDM comparison ---
    have_lcdm = args.H0 is not None and args.Omega_m0 is not None
    if have_lcdm:
        H_lcdm, q_lcdm, j_lcdm = lcdm_H_q_j(
            z, args.H0, args.Omega_m0, args.Omega_r0
        )
    else:
        H_lcdm = q_lcdm = j_lcdm = None

    # --- Plot H(z) ---
    plt.figure()
    plt.plot(z, H_tfgr, label="TFGR H(z)")
    if have_lcdm:
        plt.plot(z, H_lcdm, linestyle="--", label="ΛCDM H(z)")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("H(z) diagnostics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png_H = f"{args.out_prefix}_H_z.png"
    plt.savefig(out_png_H, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Phase 183] Saved H(z) plot -> {out_png_H}")

    # --- Plot q(z) ---
    plt.figure()
    plt.plot(z, q_tfgr, label="TFGR q(z)")
    if have_lcdm:
        plt.plot(z, q_lcdm, linestyle="--", label="ΛCDM q(z)")
    plt.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    plt.xlabel("z")
    plt.ylabel("q(z)")
    plt.title("Deceleration parameter q(z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png_q = f"{args.out_prefix}_q_z.png"
    plt.savefig(out_png_q, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Phase 183] Saved q(z) plot -> {out_png_q}")

    # --- Plot j(z) ---
    plt.figure()
    plt.plot(z, j_tfgr, label="TFGR j(z)")
    if have_lcdm:
        # ΛCDM jerk = 1
        plt.axhline(1.0, linestyle="--", label="ΛCDM j(z)=1")
    plt.xlabel("z")
    plt.ylabel("j(z)")
    plt.title("Jerk parameter j(z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png_j = f"{args.out_prefix}_j_z.png"
    plt.savefig(out_png_j, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Phase 183] Saved j(z) plot -> {out_png_j}")

    print("=== Phase 183: done ===")


if __name__ == "__main__":
    main()
