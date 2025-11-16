#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase51_llr_multi_analysis.py
-----------------------------------
Phase 51-D : Multi-reflector LLR TFGR Analysis
(Apollo 11 / 14 / 15)

複数の LLR 反射器 CSV を同時に解析し、
Δt(L) 比較プロットおよび 3D 時間場トモグラフィーを作成する。

入力CSVフォーマット:
    time_utc, station, reflector, L_m, dt_res_s, distance_km_geom

出力:
    phase51D_llr_compare_dt_vs_L.png
    phase51D_llr_compare_L_dt_Phi3D.png
    phase51D_llr_compare_summary.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

C_LIGHT = 299792458.0  # m/s


def tfgr_model(L, Lc, p, q, dt0=1.0):
    """TFGR時間補正関数 Δt(L)/Δt0"""
    return dt0 * (1.0 + (L / Lc) ** p) ** q


def analyze_reflector(df, Lc, p, q):
    """1反射器あたりの統計値算出"""
    L = df["L_m"].values
    dt_obs = df["dt_res_s"].values
    dt_model = tfgr_model(L, Lc, p, q, dt_obs.min())
    resid = dt_obs - dt_model
    chi2 = np.sum(resid**2)
    return {"chi2": chi2, "N": len(L)}


def main():
    parser = argparse.ArgumentParser(description="Multi-reflector TFGR analysis")
    parser.add_argument("--csv", required=True,
                        help="カンマ区切りのCSVファイル一覧")
    parser.add_argument("--out", default="phase51D_llr_compare",
                        help="出力ファイル名のベース")
    parser.add_argument("--Lc", type=float, default=4.0e9)
    parser.add_argument("--p", type=float, default=0.21)
    parser.add_argument("--q", type=float, default=1.32)
    args = parser.parse_args()

    csv_list = [c.strip() for c in args.csv.split(",")]
    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    fig, ax = plt.subplots(figsize=(8, 6))

    summary_records = []

    for csv_file, color in zip(csv_list, colors):
        df = pd.read_csv(csv_file)
        refl = df["reflector"].iloc[0] if "reflector" in df.columns else csv_file
        color = next(colors)
        ax.scatter(df["L_m"], df["dt_res_s"], label=refl, s=16, color=color, alpha=0.7)

        # モデル曲線
        L_range = np.linspace(df["L_m"].min(), df["L_m"].max(), 300)
        dt_model = tfgr_model(L_range, args.Lc, args.p, args.q, df["dt_res_s"].min())
        ax.plot(L_range, dt_model, color=color, linestyle="--")

        stats = analyze_reflector(df, args.Lc, args.p, args.q)
        summary_records.append({
            "reflector": refl,
            "N_data": stats["N"],
            "chi2_tfgr": stats["chi2"],
            "Lc_m": args.Lc,
            "p": args.p,
            "q": args.q,
        })

    # ---- Δt vs L plot ----
    ax.set_xlabel("L [m]")
    ax.set_ylabel("Δt_res [s]")
    ax.set_title("Phase 51-D: LLR Reflectors Δt(L) Comparison (TFGR)")
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{args.out}_dt_vs_L.png", dpi=200)
    print(f"✅ Δt–L プロット出力: {args.out}_dt_vs_L.png")

    # ---- 3D Φt plot ----
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig3d = plt.figure(figsize=(8, 6))
    ax3 = fig3d.add_subplot(111, projection="3d")

    for csv_file, color in zip(csv_list, cycle(["#1f77b4", "#ff7f0e", "#2ca02c"])):
        df = pd.read_csv(csv_file)
        refl = df["reflector"].iloc[0]
        Φt = C_LIGHT**2 * df["dt_res_s"] / df["dt_res_s"].min()
        ax3.scatter(df["L_m"], df["dt_res_s"], Φt, s=16, label=refl, color=color, alpha=0.7)

    ax3.set_xlabel("L [m]")
    ax3.set_ylabel("Δt_res [s]")
    ax3.set_zlabel("Φₜ [J/kg]")
    ax3.set_title("Phase 51-D: Time-field Tomography (Φₜ–L–Δt)")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}_L_dt_Phi3D.png", dpi=200)
    print(f"✅ 3D 時間場トモグラフィー出力: {args.out}_L_dt_Phi3D.png")

    # ---- summary table ----
    df_sum = pd.DataFrame(summary_records)
    df_sum.to_csv(f"{args.out}_summary.csv", index=False)
    print(f"✅ サマリ出力: {args.out}_summary.csv")
    print(df_sum)


if __name__ == "__main__":
    main()
