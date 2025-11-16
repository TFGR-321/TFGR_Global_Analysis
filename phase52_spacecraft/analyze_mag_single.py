# ==========================================================
# analyze_mag_single.py
# Rosetta RPC-MAG等の磁場観測データを解析し、
# 時刻・磁場ベクトル・合成磁場強度(B_total)を抽出する。
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# 設定
# ==========================================================
INPUT_FILE = "mag_der_sc_ib_a001_e2k_00000_20190306.tab.txt"
OUTPUT_CSV = "output/mag_data_processed.csv"
OUTPUT_FIG = "output_mag_components.png"

# ==========================================================
# 1. ファイル読み込み（カンマ区切り対応）
# ==========================================================
try:
    df = pd.read_csv(INPUT_FILE, sep=",", engine="python", header=None)
    print(f"\n=== Loaded file: {INPUT_FILE} ===")
    print(f"Shape: {df.shape}")
    print(df.head())
except Exception as e:
    print(f"❌ 読み込みエラー: {e}")
    exit()

# ==========================================================
# 2. 時刻列の検出と変換
# ==========================================================
time_col = 0
try:
    df["time"] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=["time"])
    print(f"\n✅ 時刻列を検出: column {time_col}")
except Exception as e:
    print(f"❌ 時刻変換エラー: {e}")
    exit()

# ==========================================================
# 3. 磁場成分列の抽出
# ==========================================================
# 通常は Bx, By, Bz が 2〜4列目に存在する想定
bx_col, by_col, bz_col = 2, 3, 4

try:
    df["Bx"] = pd.to_numeric(df[bx_col], errors='coerce')
    df["By"] = pd.to_numeric(df[by_col], errors='coerce')
    df["Bz"] = pd.to_numeric(df[bz_col], errors='coerce')
    df["B_total"] = np.sqrt(df["Bx"]**2 + df["By"]**2 + df["Bz"]**2)
    df = df.dropna(subset=["B_total"])
    print(f"✅ 磁場成分列を [{bx_col}, {by_col}, {bz_col}] としてB_totalを計算しました。")
except Exception as e:
    print(f"❌ 磁場成分処理エラー: {e}")
    exit()

# ==========================================================
# 4. 統計出力
# ==========================================================
print("\n=== 磁場統計量（単位: nT または等価値） ===")
print(df[["B_total"]].describe())

# ==========================================================
# 5. CSV出力
# ==========================================================
os.makedirs("output", exist_ok=True)
df_out = df[["time", "Bx", "By", "Bz", "B_total"]]
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 出力ファイル: {OUTPUT_CSV} を保存しました。")

# ==========================================================
# 6. 図の作成
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["Bx"], label="Bx")
plt.plot(df["time"], df["By"], label="By")
plt.plot(df["time"], df["Bz"], label="Bz")
plt.plot(df["time"], df["B_total"], label="|B|", linewidth=2, color="black")
plt.xlabel("Time")
plt.ylabel("Magnetic field (nT or equivalent)")
plt.title("Magnetic Field Components and Total Field")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=200)
plt.close()
print(f"✅ 図: {OUTPUT_FIG} を保存しました。")

print("\n🎯 解析完了: 出力ファイルをTFGR結合解析に利用できます。")
