#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 35: TFGR Temporal Field Gradient Mapping
-----------------------------------------------
入力: tfgr_unified_scaling_curve.csv（Phase 34出力）
出力:
  - tfgr_field_gradient.png
  - tfgr_field_gradient.csv

目的:
  Δt(L) から時間場 Φₜ(L) = c² * Δt(L) / L を算出し、
  そのスケール微分 dΦₜ/dL を描画。
  → Λ項の物理的実体を時間場勾配として可視化する。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


C = 299792458.0  # 光速 [m/s]


def compute_field_gradient(df):
    """Δt(L) → Φₜ(L) とその微分を求める"""
    L = np.asarray(df["L_m"], dtype=float)
    dt = np.asarray(df["dt_tfgr"], dtype=float)

    # Φₜ = c² * Δt / L
    Phi = (C**2) * dt / L

    # 数値微分
    dPhi_dL = np.gradient(Phi, L)

    df_out = pd.DataFrame({
        "L_m": L,
        "Delta_t": dt,
        "Phi_t": Phi,
        "dPhi_dL": dPhi_dL
    })
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Phase 35: TFGR Temporal Field Gradient Mapping"
    )
    parser.add_argument(
        "--csv", type=str, default="tfgr_unified_scaling_curve.csv",
        help="Input CSV file (from Phase 34)"
    )
    parser.add_argument(
        "--out", type=str, default="tfgr_field_gradient",
        help="Output prefix"
    )
    args = parser.parse_args()

    print("=== TFGR Field Gradient Mapping ===")
    print(f"Input: {args.csv}")

    # CSV読み込み
    df = pd.read_csv(args.csv)
    df_out = compute_field_gradient(df)

    # --- Φₜ(L) プロット ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(df_out["L_m"], df_out["Phi_t"], color="black", lw=2, label="Φₜ(L)")
    ax1.set_xlabel("Distance L [m]")
    ax1.set_ylabel("Φₜ [m²/s²]")
    ax1.set_title("Temporal Field Φₜ(L)")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig(f"{args.out}_phi.png", dpi=200)
    print(f"✅ Φₜ(L) 図を保存: {args.out}_phi.png")

    # --- dΦₜ/dL プロット ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_xscale("log")
    ax2.plot(df_out["L_m"], np.abs(df_out["dPhi_dL"]),
             color="red", lw=2, label="|dΦₜ/dL|")
    ax2.set_xlabel("Distance L [m]")
    ax2.set_ylabel("|dΦₜ/dL| [m/s²]")
    ax2.set_title("Gradient of Temporal Field |dΦₜ/dL|")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()

    fig2.tight_layout()
    fig2.savefig(f"{args.out}_gradient.png", dpi=200)
    print(f"✅ 勾配図を保存: {args.out}_gradient.png")

    # --- CSV出力 ---
    df_out.to_csv(f"{args.out}.csv", index=False)
    print(f"✅ 出力CSV: {args.out}.csv")


if __name__ == "__main__":
    main()
