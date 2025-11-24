import argparse
import pandas as pd
from datetime import datetime

def read_clk(file):
    rows = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("AS "):
                continue

            parts = line.split()
            # 期待フォーマット
            # ['AS', 'G26', '2025', '11', '01', '03', '11', '00.000000', '1', '-0.xxxxxxxxE-03']

            if len(parts) < 10:
                continue

            sat = parts[1]

            year  = int(parts[2])
            month = int(parts[3])
            day   = int(parts[4])
            hour  = int(parts[5])
            minu  = int(parts[6])
            sec   = float(parts[7])

            clk_bias = float(parts[9])
            clk_sigma = 1.0  # always 1 in MGX

            time_str = f"{year:04d}/{month}/{day} {hour}:{minu:02d}"

            rows.append({
                "time": time_str,
                "sat": sat,
                "clk_bias_s": clk_bias,
                "clk_sigma": clk_sigma
            })

    df = pd.DataFrame(rows)
    print(df.head(20))
    print(f"[CLK] rows={len(df)} sats={df['sat'].nunique()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clk", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[reading CLK] {args.clk}")
    df = read_clk(args.clk)

    df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
