import argparse
import math
import pandas as pd
from datetime import datetime

# -----------------------------
# SP3 → CSV 完全対応パーサー
# -----------------------------

def read_sp3(path):
    rows = []
    curr_epoch = None

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()

            # -------------------------
            # エポック行 "*  2025 11  1  0  5  0.00000000"
            # -------------------------
            if line.startswith("*"):
                parts = line.split()
                # parts = ["*", "2025","11","1","0","5","0.00000000"]
                year  = int(parts[1])
                month = int(parts[2])
                day   = int(parts[3])
                hour  = int(parts[4])
                minu  = int(parts[5])
                sec   = float(parts[6])
                curr_epoch = datetime(year, month, day, hour, minu, int(sec), int((sec % 1)*1e6))
                continue

            # -------------------------
            # 衛星座標行 "PG01  14826.33  -21997.07 ... 354.116104"
            # -------------------------
            if line.startswith("P") and curr_epoch is not None:
                parts = line.split()

                # parts = ["PG01","14826.331865","-21997.071279","655.099040","354.116104"]
                sat = parts[0]       # PG01, PR02, PE07, PC08, PJ03 etc.
                x_km = float(parts[1])
                y_km = float(parts[2])
                z_km = float(parts[3])
                clk_bias_ns = float(parts[4])  # ns

                x_m = x_km * 1000.0
                y_m = y_km * 1000.0
                z_m = z_km * 1000.0
                L_m = math.sqrt(x_m*x_m + y_m*y_m + z_m*z_m)
                clk_s = clk_bias_ns * 1e-9     # ns → s

                rows.append([
                    curr_epoch, sat, x_m, y_m, z_m, L_m, clk_s
                ])

    df = pd.DataFrame(rows, columns=["time","sat","x_m","y_m","z_m","L_m","clk_bias_s"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[reading SP3] {args.sp3}")
    df = read_sp3(args.sp3)

    print(df.head())
    print(f"rows={len(df)} sats={df['sat'].nunique()}")

    df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
