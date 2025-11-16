#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase51E_tfgr_global_tomography.py
-----------------------------------
Phase 51-E : Earth–Satellite–Moon time-field tomography

入力:
  GPS側:  time_utc, station, sat, L_m, dt_res_s, ...
  LLR側:  time_utc, station, reflector, L_m, dt_res_s, ...

出力:
  <out>_dt_vs_L_global.png
  <out>_L_dt_Phi3D_global.png
  <out>_summary_global.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa

C_LIGHT = 299792458.0  # m/s


def tfgr_model(L, Lc, p, q, dt0):
    """TFGR 時間補正関数 Δt(L)"""
    return dt0 * (1.0 + (L / Lc) ** p) ** q


def load_gps_csv(path):
    df = pd.read_csv(path)
    if "L_m" not in df.columns or "dt_res_s" not in df.columns:
        raise ValueError(f"GPS CSV に L_m / dt_res_s 列がありません: {path}")
    station = df["station"].iloc[0] if "station" in df.columns else "GPS"
    df["dataset"] = f"GPS_{station}"
    return df


def load_llr_csv(path):
    df = pd.read_csv(path)
    if "L_m" not in df.columns or "dt_res_s" not in df.columns:
        raise ValueError(f"LLR CSV に L_m / dt_res_s 列がありません: {path}")
    if "reflector" in df.columns:
        refl = df["reflector"].iloc[0]
    else:
        refl = "LLR"
    df["dataset"] = f"LLR_{refl}"
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Global TFGR tomography using GPS + LLR data."
    )
    parser.add_argument("--gps_csv", required=True,
                        help="GPS側 CSV ファイル（例: AJAC_phase51B.csv）")
    parser.add_argument("--llr_csv", required=True,
                        help="カンマ区切りの LLR CSV ファイル一覧")
    parser.add_argument("--out", default="phase51E_global",
                        help="出力ファイル名のベース")
    parser.add_argument("--Lc", type=float, default=4.0e9)
    parser.add_argument("--p", type=float, default=0.21)
    parser.add_argument("--q", type=float, default=1.32)
    args = parser.parse_args()

    # ---- データ読み込み ----
    gps_df = load_gps_csv(args.gps_csv)

    llr_paths = [s.strip() for s in args.llr_csv.split(",") if s.strip()]
    llr_dfs = [load_llr_csv(p) for p in llr_paths]

    all_df = pd.concat([gps_df] + llr_dfs, ignore_index=True)

    # グローバル基準Δt0（全データ中の最小値）
    dt0 = all_df["dt_res_s"].min()
    L_min = all_df["L_m"].min()
    L_max = all_df["L_m"].max()

    print(f"全データ点数: {len(all_df)}")
    print(f"Δt0 (min dt_res_s) = {dt0:.3e} s")
    print(f"L 範囲: {L_min:.3e} – {L_max:.3e} m")

    # ---- Δt vs L (グローバル) ----
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, df_sub in all_df.groupby("dataset"):
        ax.scatter(df_sub["L_m"], df_sub["dt_res_s"], s=18, alpha=0.8, label=name)

    L_range = np.linspace(L_min, L_max, 500)
    dt_model = tfgr_model(L_range, args.Lc, args.p, args.q, dt0)
    ax.plot(L_range, dt_model, "k--", label="TFGR model (global)")

    ax.set_xlabel("L [m] (geocentric distance)")
    ax.set_ylabel("Clock residual Δt [s]")
    ax.set_title("Phase 51-E: GPS + LLR global Δt(L) (TFGR)")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.out}_dt_vs_L_global.png", dpi=200)
    print(f"✅ Δt–L プロット出力: {args.out}_dt_vs_L_global.png")

    # ---- 3D Φ_t トモグラフィー ----
    fig3d = plt.figure(figsize=(8, 6))
    ax3 = fig3d.add_subplot(111, projection="3d")

    for name, df_sub in all_df.groupby("dataset"):
        Phi_t = C_LIGHT ** 2 * df_sub["dt_res_s"] / dt0
        ax3.scatter(df_sub["L_m"], df_sub["dt_res_s"], Phi_t,
                    s=18, alpha=0.8, label=name)

    ax3.set_xlabel("L [m]")
    ax3.set_ylabel("Δt_res [s]")
    ax3.set_zlabel("Φₜ [J/kg] (relative)")
    ax3.set_title("Phase 51-E: Vertical time-field tomography (Earth–Moon)")
    ax3.legend()
    plt.tight_layout()
    fig3d.savefig(f"{args.out}_L_dt_Phi3D_global.png", dpi=200)
    print(f"✅ 3D 時間場トモグラフィー出力: {args.out}_L_dt_Phi3D_global.png")

    # ---- サマリー ----
    records = []
    for name, df_sub in all_df.groupby("dataset"):
        records.append(
            {
                "dataset": name,
                "N_data": len(df_sub),
                "L_min_m": df_sub["L_m"].min(),
                "L_max_m": df_sub["L_m"].max(),
                "dt_min_s": df_sub["dt_res_s"].min(),
                "dt_max_s": df_sub["dt_res_s"].max(),
            }
        )
    df_sum = pd.DataFrame(records)
    df_sum.to_csv(f"{args.out}_summary_global.csv", index=False)
    print(f"✅ サマリ出力: {args.out}_summary_global.csv")
    print(df_sum)


if __name__ == "__main__":
    main()
