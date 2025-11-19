#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
phase32_timefield_tomography.py

Phase 32 – Time-Field Tomography (GPS Time-Field Map)

- Take multiple Phase 31 summary CSVs (AJAC / ALIC / ANK2 / MIZU).
- Combine them into a single dataframe.
- Filter to "Gxx" satellites (G01–G32) with enough N_points.
- Compute per-station statistics:
    * mean_A_sat, std_A_sat, median_A_sat
    * mean_R2, mean_delta_AIC, N_sats_used
- Optionally merge station latitude/longitude from station_info.csv.
- Output:
    * station-level stats CSV
    * A_sat pivot table (sat × station)
    * (optional) latitude vs mean_A_sat plot
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_concat_summaries(summary_files):
    """Load multiple Phase31 summary CSVs and concatenate."""
    frames = []
    for path in summary_files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise ValueError("No summary files loaded.")
    return pd.concat(frames, ignore_index=True)


def filter_satellite_rows(df, min_points=3):
    """
    Keep only:
      - sat starting with 'G' (G01–G32)
      - N_points >= min_points
      - finite A_sat
    """
    df = df.copy()
    # Ensure sat is string
    df["sat"] = df["sat"].astype(str)
    mask = (
        df["sat"].str.startswith("G") &
        (df["N_points"] >= min_points) &
        np.isfinite(df["A_sat"])
    )
    return df[mask].reset_index(drop=True)


def compute_station_stats(df):
    """
    Compute station-level statistics from per-sat rows.
    Returns a dataframe with one row per station.
    """
    def _agg(series):
        return {
            "mean_A_sat": series["A_sat"].mean(),
            "std_A_sat": series["A_sat"].std(ddof=1),
            "median_A_sat": series["A_sat"].median(),
            "mean_R2": series["R2"].mean(),
            "mean_delta_AIC": series["delta_AIC"].mean(),
            "N_sats_used": len(series)
        }

    grouped = df.groupby("station").apply(_agg)
    # grouped は Series of dict -> DataFrame に整形
    stats_df = pd.DataFrame(list(grouped.values), index=grouped.index)
    stats_df.index.name = "station"
    stats_df = stats_df.reset_index()
    return stats_df


def merge_station_info(stats_df, info_path=None):
    """Merge station latitude/longitude if given."""
    if info_path is None:
        return stats_df

    info_df = pd.read_csv(info_path)
    # station 列でマージ（inner merge）
    merged = stats_df.merge(info_df, on="station", how="left")
    return merged


def make_A_pivot(df, out_path):
    """Create sat × station pivot table for A_sat and save as CSV."""
    pivot = df.pivot_table(
        index="sat",
        columns="station",
        values="A_sat"
    )
    pivot.to_csv(out_path)
    return pivot


def plot_lat_vs_A(stats_df, out_path):
    """
    Plot latitude vs mean_A_sat if lat_deg is available.
    """
    if "lat_deg" not in stats_df.columns:
        print("[INFO] No 'lat_deg' column, skip latitude plot.")
        return

    df = stats_df.dropna(subset=["lat_deg", "mean_A_sat"])
    if df.empty:
        print("[INFO] No valid rows for latitude plot.")
        return

    plt.figure()
    plt.scatter(df["lat_deg"], df["mean_A_sat"])
    for _, row in df.iterrows():
        label = row["station"]
        plt.text(row["lat_deg"], row["mean_A_sat"], f" {label}", fontsize=8)

    plt.xlabel("Latitude [deg]")
    plt.ylabel("Mean A_sat")
    plt.title("Time-Field Strength vs Latitude (Phase 32)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved latitude plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 32 – Time-Field Tomography (GPS Time-Field Map)"
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="Phase31 summary CSV files (e.g. phase31_AJAC_summary.csv ...)"
    )
    parser.add_argument(
        "--station_info",
        default=None,
        help="Optional station info CSV with columns: station,lat_deg,lon_deg,height_m"
    )
    parser.add_argument(
        "--out_prefix",
        default="phase32_timefield_tomography",
        help="Prefix for output files."
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Output directory."
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=3,
        help="Minimum N_points per satellite to be used. Default: 3"
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading Phase31 summaries...")
    df_all = load_and_concat_summaries(args.summaries)

    print("[INFO] Filtering satellite rows (Gxx, N_points >= {0})...".format(args.min_points))
    df_g = filter_satellite_rows(df_all, min_points=args.min_points)

    if df_g.empty:
        print("[WARN] No valid rows after filtering. Check inputs.")
        return

    # Save filtered per-satellite table (for record)
    filtered_path = os.path.join(args.out_dir, f"{args.out_prefix}_per_satellite_filtered.csv")
    df_g.to_csv(filtered_path, index=False)
    print(f"[INFO] Saved filtered per-satellite data to: {filtered_path}")

    # Station-level statistics
    print("[INFO] Computing station-level stats...")
    station_stats = compute_station_stats(df_g)
    station_stats = merge_station_info(station_stats, args.station_info)

    stats_path = os.path.join(args.out_dir, f"{args.out_prefix}_station_stats.csv")
    station_stats.to_csv(stats_path, index=False)
    print(f"[INFO] Saved station stats to: {stats_path}")

    # A_sat pivot table (sat × station)
    pivot_path = os.path.join(args.out_dir, f"{args.out_prefix}_A_sat_pivot.csv")
    make_A_pivot(df_g, pivot_path)
    print(f"[INFO] Saved A_sat pivot to: {pivot_path}")

    # Plot latitude vs mean_A_sat if possible
    lat_plot_path = os.path.join(args.out_dir, f"{args.out_prefix}_lat_vs_mean_A.png")
    plot_lat_vs_A(station_stats, lat_plot_path)


if __name__ == "__main__":
    main()
