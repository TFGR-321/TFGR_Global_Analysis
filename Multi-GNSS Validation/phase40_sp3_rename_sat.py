import pandas as pd
import argparse

def convert_sp3_sat(sp3_csv, out_csv):
    print(f"[reading] {sp3_csv}")
    df = pd.read_csv(sp3_csv)

    def rename_sat(s):
        # SP3は PGxx / PRxx / PExx / PCxx / PJxx のように P + 系統 + 番号
        # → CLK の Gxx / Rxx / Exx / Cxx / Jxx に変換
        if isinstance(s, str) and len(s) >= 4 and s[0] == "P":
            return s[1:]  # 先頭の "P" を削除
        return s

    df["sat"] = df["sat"].apply(rename_sat)

    print(df.head(10))
    print(f"[saving] {out_csv}")
    df.to_csv(out_csv, index=False)
    print("[done]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp3_csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    convert_sp3_sat(args.sp3_csv, args.out)
