import pandas as pd
import argparse
from datetime import datetime, timedelta

def merge_sp3_clk(sp3_csv, clk_csv, out_csv, tol_s=300):
    print(f"[reading SP3] {sp3_csv}")
    df_sp3 = pd.read_csv(sp3_csv)
    df_sp3["time"] = pd.to_datetime(df_sp3["time"])

    print(f"[reading CLK] {clk_csv}")
    df_clk = pd.read_csv(clk_csv)
    df_clk["time"] = pd.to_datetime(df_clk["time"])

    # 出力リスト
    merged_rows = []

    # 衛星ごとに処理
    for sat in sorted(df_sp3["sat"].unique()):
        df_sp = df_sp3[df_sp3["sat"] == sat]
        df_ck = df_clk[df_clk["sat"] == sat]

        if df_ck.empty:
            continue

        for _, row in df_sp.iterrows():
            t_sp = row["time"]

            # 時刻差（秒）
            df_ck["dt"] = (df_ck["time"] - t_sp).abs().dt.total_seconds()

            # 最も近い行
            row_ck = df_ck.loc[df_ck["dt"].idxmin()]

            # 許容範囲外ならスキップ
            if row_ck["dt"] > tol_s:
                continue

            merged_rows.append({
                "time": t_sp,
                "sat": sat,
                "x_m": row["x_m"],
                "y_m": row["y_m"],
                "z_m": row["z_m"],
                "L_m": row["L_m"],
                "clk_bias_s": row_ck["clk_bias_s"]
            })

    df_out = pd.DataFrame(merged_rows)
    print(df_out.head())
    print(f"[merged rows] {len(df_out)}")
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp3_csv", required=True)
    parser.add_argument("--clk_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tol_s", type=float, default=300)
    args = parser.parse_args()

    merge_sp3_clk(args.sp3_csv, args.clk_csv, args.out, args.tol_s)
