#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import re
from pathlib import Path

def is_gps_sat(s):
    """
    sat名が "G01"〜"G32" などGPS形式か判定。
    文字列の先頭がGで、その後に数字2桁が続くものをGPSとする。
    """
    if not isinstance(s, str):
        return False
    return re.match(r"^G\d{2}$", s.strip()) is not None

def main():
    ap = argparse.ArgumentParser(
        description="Filter merged orbit-clock CSV to GPS-only sats (Gxx)."
    )
    ap.add_argument("--in_csv", required=True, help="Merged orbit-clock CSV (e.g., phase41_merged_orbit_clock.csv)")
    ap.add_argument("--out_csv", required=True, help="Output GPS-only CSV")
    ap.add_argument("--also_save_non_gps", action="store_true",
                    help="If set, also save non-GPS rows as <out_csv>_non_gps.csv")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)

    # 必須列チェック
    required_cols = {"time", "sat", "x_m", "y_m", "z_m", "L_m", "clk_bias_s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # GPSのみ抽出
    gps_mask = df["sat"].apply(is_gps_sat)
    df_gps = df[gps_mask].copy()
    df_gps.sort_values(["sat", "time"], inplace=True)

    # 出力
    df_gps.to_csv(out_path, index=False)

    print(f"[GPS only] rows={len(df_gps)} sats={df_gps['sat'].nunique()}")
    print(f"[saved] {out_path}")

    if args.also_save_non_gps:
        df_non = df[~gps_mask].copy()
        non_path = out_path.with_name(out_path.stem + "_non_gps.csv")
        df_non.to_csv(non_path, index=False)
        print(f"[non-GPS] rows={len(df_non)} sats={df_non['sat'].nunique()}")
        print(f"[saved] {non_path}")

if __name__ == "__main__":
    main()
