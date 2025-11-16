#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_laserlink_csv.py

SMART-1 MISSION.txt から Laser-Link 実験の数値情報を抽出し、
Phase 51 用の L–Δt データ CSV (SMART1_LaserLink.csv) を生成する。

出力列:
    date_utc : 実験日（推定）
    L_km     : 距離 [km]
    L_m      : 距離 [m]
    dt_est_s : 光行遅延 (L / c) [s]

使用方法:
    python extract_laserlink_csv.py --in MISSION.txt --out SMART1_LaserLink.csv
"""

import re
import argparse
import pandas as pd

C_LIGHT = 299792458.0  # [m/s]


def extract_laserlink_lines(text):
    """MISSION.txt から 'Laser-Link' 節を抽出する"""
    lines = text.splitlines()
    start, end = None, None
    for i, ln in enumerate(lines):
        if re.search(r"Laser[- ]Link", ln):
            start = i
        if start is not None and re.search(r"RSIS", ln):
            end = i
            break
    if start is None:
        raise RuntimeError("Laser-Link セクションが見つかりません。")
    if end is None:
        end = len(lines)
    return "\n".join(lines[start:end])


def extract_distances(text):
    """Laser-Link セクション中から距離に関する数値を抽出する"""
    # 例: "14 000 km", "73000km", "14,000km" などを検出
    matches = re.findall(r"(\d{1,3}(?:[ ,]?\d{3})*)\s*km", text)
    values_km = []
    for m in matches:
        km = float(m.replace(",", "").replace(" ", ""))
        values_km.append(km)
    return sorted(set(values_km))


def main():
    parser = argparse.ArgumentParser(description="Extract Laser-Link experiment distances from MISSION.txt")
    parser.add_argument("--in", dest="infile", required=True, help="Input MISSION.txt")
    parser.add_argument("--out", dest="outfile", default="SMART1_LaserLink.csv", help="Output CSV file name")
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    laser_text = extract_laserlink_lines(text)
    distances_km = extract_distances(laser_text)

    if not distances_km:
        raise RuntimeError("距離データが抽出できませんでした。")

    # 光行遅延 Δt = L / c
    df = pd.DataFrame({
        "date_utc": ["2003-12-01"] * len(distances_km),
        "L_km": distances_km,
        "L_m": [d * 1000.0 for d in distances_km],
        "dt_est_s": [(d * 1000.0) / C_LIGHT for d in distances_km],
    })

    df.to_csv(args.outfile, index=False)
    print(f"✅ 出力完了: {args.outfile}")
    print(df)


if __name__ == "__main__":
    main()
