import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tfgr_dt(L_m, Lc, p, q, dt0=1.0):
    """時間場補正式 Δt(L)"""
    return dt0 * (1.0 + (L_m / Lc)**p)**q

def main():
    parser = argparse.ArgumentParser(
        description="TFGR correction analysis using Voyager geometry"
    )
    parser.add_argument("--csv", required=True, help="Voyager 距離データCSV")
    parser.add_argument("--Lc", type=float, default=4.0e9, help="臨界長スケール [m]")
    parser.add_argument("--p", type=float, default=0.21, help="スケーリング指数 p")
    parser.add_argument("--q", type=float, default=1.32, help="スケーリング指数 q")
    parser.add_argument("--out", required=True, help="出力ベース名")
    args = parser.parse_args()

    # CSV 読み込み
    df = pd.read_csv(args.csv)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    print("Loaded:", len(df), "rows")

    # TFGR Δt 計算
    df["dt_res"] = tfgr_dt(df["L_m"], args.Lc, args.p, args.q)

    # 速度推定
    df["v_km_s"] = df["L_km"].diff() / df["time_utc"].diff().dt.total_seconds()

    # 結果保存
    out_csv = f"{args.out}_tfgr.csv"
    df.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)

    # Δt vs L
    plt.figure(figsize=(7,4))
    plt.plot(df["L_km"], df["dt_res"], ".", markersize=3)
    plt.xlabel("Distance L [km]")
    plt.ylabel("Δt_res [s]")
    plt.title("TFGR Correction vs Distance (Voyager 1 – Jupiter)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out}_tfgr_vs_L.png", dpi=200)
    plt.close()

    # Δt vs Time
    plt.figure(figsize=(7,4))
    plt.plot(df["time_utc"], df["dt_res"], color="red")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Δt_res [s]")
    plt.title("Temporal Evolution of TFGR Correction (Voyager 1 – Jupiter)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out}_tfgr_vs_time.png", dpi=200)
    plt.close()

    # v(t)
    plt.figure(figsize=(7,4))
    plt.plot(df["time_utc"], df["v_km_s"], color="blue")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Relative Velocity Profile (Voyager 1 – Jupiter)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out}_velocity.png", dpi=200)
    plt.close()

    print("✅ Plots saved successfully.")

if __name__ == "__main__":
    main()
