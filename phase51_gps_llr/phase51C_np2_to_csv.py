#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase51C_np2_to_csv.py
-----------------------------------
ILRS / NASA Lunar Laser Ranging Normal Point ファイル (.np2 / .npt)
→ Phase 51-C / D 用 TFGR入力 CSV に変換する。

入力:
    apollo11_20250627.np2.txt
    apollo14_202012.np2.txt
    apollo15_202502.np2.txt
出力:
    apollo11_20250627_phase51D.csv
    apollo14_202012_phase51D.csv
    apollo15_202502_phase51D.csv

出力列:
    time_utc, station, reflector, L_m, dt_res_s, distance_km_geom

処理概要:
    ・h2 → station 名
    ・h3 → reflector 名
    ・h4 → 日付
    ・11 → 時刻(秒 of day)＋往復光時間[ps]
    ・20 → 幾何距離[km]
    ・往復→片道換算：tof/2
"""

import argparse
import os
import re
import datetime as dt
import pandas as pd

C_LIGHT = 299792458.0  # m/s


def parse_np2_file(filepath):
    records = []
    station, reflector = None, None
    current_date = None
    current_distance_km = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            tag = line[:2].lower()
            parts = line[2:].strip().split()

            # 観測局
            if tag == "h2" and len(parts) > 0:
                station = parts[0]

            # ターゲット（月面反射器）
            elif tag == "h3" and len(parts) > 0:
                reflector = parts[0]

            # 観測日 (h4)
            elif tag == "h4" and len(parts) >= 4:
                try:
                    year = int(parts[1])
                    month = int(parts[2])
                    day = int(parts[3])
                    current_date = dt.date(year, month, day)
                except Exception:
                    current_date = None

            # 幾何データ (20)
            elif tag == "20":
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(nums) > 0:
                    current_distance_km = float(nums[0])

            # 観測データ (11)
            elif tag == "11" and len(parts) >= 2:
                try:
                    sod = float(parts[0])  # 秒 of day
                    tof_ps = float(parts[1])  # time-of-flight [ps]
                except ValueError:
                    continue

                if current_date is None:
                    continue

                tof_s = tof_ps * 1e-12
                dt_oneway = tof_s / 2.0
                L_m = C_LIGHT * dt_oneway

                base_dt = dt.datetime(
                    current_date.year, current_date.month, current_date.day
                )
                obs_dt = base_dt + dt.timedelta(seconds=sod)
                time_utc = obs_dt.isoformat() + "Z"

                records.append(
                    {
                        "time_utc": time_utc,
                        "station": station,
                        "reflector": reflector,
                        "L_m": L_m,
                        "dt_res_s": dt_oneway,
                        "distance_km_geom": current_distance_km,
                    }
                )

    if not records:
        raise RuntimeError(f"No valid '11' records found in {filepath}")

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert ILRS LLR np2/npt files to TFGR CSV format."
    )
    parser.add_argument(
        "--in",
        dest="infiles",
        nargs="+",
        required=True,
        help="入力ファイル（複数指定可）例: --in apollo11_20250627.np2.txt apollo14_202012.np2.txt",
    )
    args = parser.parse_args()

    for infile in args.infiles:
        df = parse_np2_file(infile)

        basename = os.path.basename(infile)
        name, _ = os.path.splitext(basename)
        outfile = f"{name}_phase51D.csv"

        df.to_csv(outfile, index=False)

        print(f"\n✅ 出力完了: {outfile}")
        print(f"  レコード数: {len(df)}")
        print(f"  観測局: {df['station'].unique().tolist()}")
        print(f"  反射器: {df['reflector'].unique().tolist()}")
        print(
            f"  距離範囲: {df['L_m'].min():.3e} – {df['L_m'].max():.3e} m"
        )


if __name__ == "__main__":
    main()
