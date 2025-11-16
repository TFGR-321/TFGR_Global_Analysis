import pandas as pd
import numpy as np
import re

def read_orbit_table(path):
    """Rosetta ICA_ELEVATION_TABLE_V02.TAB.txtを読み込み、time列を生成"""
    df = pd.read_csv(path, header=None, sep=",", engine="python", comment="#")
    print(f"[orbit] raw shape = {df.shape}")

    # 先頭列が0,1,2,...ならこれは行番号。除外。
    if df.iloc[0, 0] == 0:
        df = df.iloc[:, 1:]

    # 仮の列名を付与
    df.columns = [f"col{i}" for i in range(df.shape[1])]
    print(f"[orbit] columns = {df.columns.tolist()[:8]} ...")

    # time列を推定
    # SC_CLOCK_START_COUNT の代わりに col0 を時間として扱う（仮）
    df["time"] = np.arange(len(df)) * 60  # 仮に1分間隔の観測とする
    # ダミー座標（後でRPCICAのX,Y,Zとマージ可能）
    df["X"] = np.linspace(-3e6, 3e6, len(df))
    df["Y"] = np.linspace(-2e6, 2e6, len(df))
    df["Z"] = np.linspace(-1e6, 1e6, len(df))
    return df

def read_rpcica_table(path):
    """RPCICA160903T00_000_L4_DE.TAB.txtを読み込み"""
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        lines = [l.strip() for l in f if re.match(r"^[0-9]", l)]
    if not lines:
        raise ValueError("No numeric lines found in RPCICA file")
    data = [re.split(r"[ ,]+", l) for l in lines]
    df = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    df.columns = [f"col{i}" for i in range(df.shape[1])]
    df["time"] = np.arange(len(df)) * 60  # 仮: 同様に1分刻み
    return df

def merge_rpcica_with_orbit(ica_df, orbit_df):
    """time近傍マージ"""
    merged = pd.merge_asof(
        ica_df.sort_values("time"),
        orbit_df.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=600  # ±10分以内
    )
    merged["L"] = np.sqrt(merged["X"]**2 + merged["Y"]**2 + merged["Z"]**2)
    print(f"[merge] done, rows = {len(merged)}")
    return merged.dropna(subset=["L"])

if __name__ == "__main__":
    orbit = read_orbit_table("ICA_ELEVATION_TABLE_V02.TAB.txt")
    ica    = read_rpcica_table("RPCICA160903T00_000_L4_DE.TAB.txt")
    merged = merge_rpcica_with_orbit(ica, orbit)
    merged.to_csv("rpcica_with_xyz.csv", index=False)
    print("[✔] rpcica_with_xyz.csv exported successfully")
