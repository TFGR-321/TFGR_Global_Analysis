import pandas as pd
import numpy as np

input_file = "rpcica_with_xyz.csv"
output_file = "rpcica_tfgr_ready_v2.csv"

df = pd.read_csv(input_file)

# --- 1. Y系列（実際に変動している信号）を使う ---
y_cols = [c for c in df.columns if c.startswith("col") and "_y" in c]

if not y_cols:
    raise RuntimeError("col*_y 列が見つからないので、dt_res を作れません。")

# 999.9 は「欠損値」の可能性が高いので NaN 扱いにする
signal = df[y_cols].replace(999.9, np.nan)

# 各時刻でエネルギー方向に合計カウントをとる
sum_signal = signal.sum(axis=1)

# 平均からのズレ（変動成分）だけを取り出す
sum_centered = sum_signal - sum_signal.mean()

# --- 2. スケールを「秒」にマッピング ---
# 最大振幅が ~1e-12 秒程度になるように線形スケーリング
max_abs = np.nanmax(np.abs(sum_centered))
if max_abs == 0 or np.isnan(max_abs):
    raise RuntimeError("信号に変動がありません（全部同じ値 or NaN）")

scale = 1e-12 / max_abs   # 最大で約 1e-12 s になるよう調整
df["dt_res"] = sum_centered * scale

# --- 3. 出力列をまとめる ---
cols_to_keep = ["time", "dt_res", "X", "Y", "Z"]
if "L" in df.columns:
    cols_to_keep.append("L")

out = df[cols_to_keep].dropna()

out.to_csv(output_file, index=False)
print(f"[✔] 出力完了: {output_file}")
print(out.head())
print(out["dt_res"].describe())
