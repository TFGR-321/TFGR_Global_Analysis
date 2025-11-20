import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def tfgr_dt(L, Lc, p, q, dt0=1.0):
    return dt0 * (1 + (L / Lc) ** p) ** q

def main():
    parser = argparse.ArgumentParser(description="Analyze TFGR correction for New Horizons–Arrokoth flyby")
    parser.add_argument("--csv", required=True, help="Input distance CSV (from phase31)")
    parser.add_argument("--Lc", type=float, required=True, help="Critical length scale [m]")
    parser.add_argument("--p", type=float, required=True, help="TFGR parameter p")
    parser.add_argument("--q", type=float, required=True, help="TFGR parameter q")
    parser.add_argument("--out", default="output_tfgr", help="Output prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "L_m" not in df.columns:
        df["L_m"] = df["L_km"] * 1e3

    # 🟢 修正：文字列から datetime に変換
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    print("=== Loaded data ===")
    print(df.head())

    # compute delta-t correction
    df["dt_tfgr"] = tfgr_dt(df["L_m"], args.Lc, args.p, args.q, dt0=1.0)
    df["dt_res"] = df["dt_tfgr"] - 1.0

    # 🟢 修正：datetime 差分を安全に計算
    df["time_diff_s"] = df["time_utc"].diff().dt.total_seconds()
    df["v_km_s"] = df["L_km"].diff() / df["time_diff_s"]
    df["v_km_s"] = df["v_km_s"].fillna(0)

    # save CSV
    out_csv = f"{args.out}.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved processed file: {out_csv}")

    # Plot Δt vs L
    plt.figure(figsize=(8,5))
    plt.plot(df["L_km"], df["dt_res"], "o", markersize=2)
    plt.xlabel("Distance L [km]")
    plt.ylabel("Δt_res [s]")
    plt.title("TFGR Correction vs Distance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{args.out}_dt_vs_L.png", dpi=200)
    plt.close()

    # Plot Δt vs Time
    plt.figure(figsize=(8,5))
    plt.plot(df["time_utc"], df["dt_res"], color="red")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Δt_res [s]")
    plt.title("Temporal Evolution of TFGR Correction")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{args.out}_dt_vs_time.png", dpi=200)
    plt.close()

    # Plot velocity
    plt.figure(figsize=(8,5))
    plt.plot(df["time_utc"], df["v_km_s"], color="blue")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Relative Velocity Profile (New Horizons – Arrokoth)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{args.out}_velocity.png", dpi=200)
    plt.close()

    print(f"✅ Figures saved as: {args.out}_*.png")

if __name__ == "__main__":
    main()
