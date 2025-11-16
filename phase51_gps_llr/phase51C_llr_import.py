#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase51C_llr_import.py
-----------------------------------
ILRS / NASA Apollo LLR Normal Point file (.np2 or .npt)
→ Phase 51-C (TFGR) 用 CSV 変換スクリプト

入力:
    apollo15_202502.np2.txt   など

出力:
    apollo15_202502_phase51C.csv

出力列:
    time_utc, station, reflector, L_m, dt_res_s

処理概要:
    ・"h3" 行 → reflector 名（例: apollo15）
    ・"h2" 行 → station 名（例: APOL, MATM）
    ・"11" 行 → range 正規点（秒 or ピコ秒単位）
    ・"20" 行 → 幾何情報（距離[km]・方位角など）
    ・L = c * tof / 2,  Δt = tof / 2 で変換
"""

import argparse
import pandas as pd
import datetime as dt
import re

C_LIGHT = 299792458.0  # m/s


def parse_llr_np2(filepath):
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

            # ターゲット（月面反射鏡）
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

            # 幾何データ（地心距離など）20
            elif tag == "20":
                # 例: 20 384400.123 km ...
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(nums) > 0:
                    current_distance_km = float(nums[0])

            # 観測正規点 (11)
            elif tag == "11" and len(parts) >= 2:
                try:
                    sod = float(parts[0])  # 秒 of day
                    tof_ps = float(parts[1])  # time of flight [ps]
                except ValueError:
                    continue

                if current_date is None:
                    continue

                # ピコ秒→秒
                tof_s = tof_ps * 1e-12
                dt_oneway = tof_s / 2.0
                L_m = C_LIGHT * dt_oneway

                obs_dt = (
                    dt.datetime(
                        current_date.year, current_date.month, current_date.day
                    )
                    + dt.timedelta(seconds=sod)
                )
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
        raise RuntimeError("No LLR normal point data found in file.")

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert LLR np2 file to TFGR CSV")
    parser.add_argument("--in", dest="infile", required=True, help="入力 np2 ファイル")
    parser.add_argument(
        "--out", dest="outfile", default="phase51C_llr_output.csv", help="出力 CSV ファイル名"
    )
    args = parser.parse_args()

    df = parse_llr_np2(args.infile)
    df.to_csv(args.outfile, index=False)

    print(f"✅ 出力完了: {args.outfile}")
    print(f"  レコード数: {len(df)}")
    print(f"  観測局: {df['station'].unique().tolist()}")
    print(f"  反射器: {df['reflector'].unique().tolist()}")
    print(f"  距離範囲: {df['L_m'].min():.3e}–{df['L_m'].max():.3e} m")


if __name__ == "__main__":
    main()
