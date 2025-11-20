# ==========================================================
# merge_tfgr_mag.py  (完全版)
# Rosetta RPC-ICA (TFGR残差) × RPC-MAG (磁場強度) 結合解析
# Phase 30B: TFGR–Magnetic Coupling
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------
# 0. 設定
# ----------------------------------------------------------
TFGR_FILE = "rpcica_tfgr_ready_v2.csv"        # Δt_res など
MAG_FILE  = "mag_data_processed.csv"   # Bx,By,Bz,B_total

OUTPUT_MERGED      = "output/tfgr_mag_merged.csv"
OUTPUT_SCATTER     = "output/tfgr_mag_scatter.png"
OUTPUT_TIMESERIES  = "output/tfgr_mag_timeseries.png"

os.makedirs("output", exist_ok=True)

print("\n=== TFGR–MAG 結合解析を開始 ===")

# ----------------------------------------------------------
# 1. データ読み込み
# ----------------------------------------------------------
print("\n=== Loading datasets ===")
df_tfgr = pd.read_csv(TFGR_FILE)
df_mag  = pd.read_csv(MAG_FILE)

print(f"TFGR rows: {len(df_tfgr)}, MAG rows: {len(df_mag)}")

# ----------------------------------------------------------
# 2. 時刻軸の整形
#    - TFGR: time = [0, 60, 120, ...]  → 「秒」だとみなす
#    - MAG : time = 実UTC (2019-03-06T...) → datetime に変換
#    - MAGの最初の時刻を t=0 として、TFGRの time(sec) をそこに乗せる
# ----------------------------------------------------------

# MAG側: datetime へ
df_mag["time"] = pd.to_datetime(df_mag["time"], errors="coerce")
df_mag = df_mag.dropna(subset=["time"])

if df_mag.empty:
    raise RuntimeError("MAGデータの time 列が全て NaT です。mag_data_processed.csv を確認してください。")

mag_start = df_mag["time"].min()
print(f"MAG start time: {mag_start}")

# TFGR側: time を「秒」として扱う
if "time" not in df_tfgr.columns:
    raise RuntimeError("TFGRファイルに 'time' 列がありません。rpcica_tfgr_ready_v2.csv を確認してください。")

df_tfgr["t_sec"] = pd.to_numeric(df_tfgr["time"], errors="coerce")
df_tfgr = df_tfgr.dropna(subset=["t_sec"])

if df_tfgr.empty:
    raise RuntimeError("TFGRデータの time(秒) が全て NaN です。rpcica_tfgr_ready_v2.csv を確認してください。")

# TFGR側の「実時間」列を作成
df_tfgr["time"] = mag_start + pd.to_timedelta(df_tfgr["t_sec"], unit="s")

print(f"TFGR time range after alignment: {df_tfgr['time'].min()} 〜 {df_tfgr['time'].max()}")
print(f"MAG  time range                : {df_mag['time'].min()} 〜 {df_mag['time'].max()}")

# 念のため NaT を除去
df_tfgr = df_tfgr.dropna(subset=["time"])

# ----------------------------------------------------------
# 3. asof merge（±30秒以内の最も近い点を対応付け）
# ----------------------------------------------------------
df_tfgr = df_tfgr.sort_values("time")
df_mag  = df_mag.sort_values("time")

merged = pd.merge_asof(
    df_tfgr,
    df_mag,
    on="time",
    direction="nearest",
    tolerance=pd.Timedelta("30s")
)

# Δt_res と B_total が両方ある行のみ利用
if "dt_res" not in merged.columns:
    raise RuntimeError("merged データに 'dt_res' 列がありません。TFGRファイルの列名を確認してください。")
if "B_total" not in merged.columns:
    raise RuntimeError("merged データに 'B_total' 列がありません。MAGファイルの列名を確認してください。")

merged = merged.dropna(subset=["dt_res", "B_total"])
print(f"\n✅ 結合後データ数: {len(merged)} 行")

if merged.empty:
    print("⚠ 結合結果が0行でした。time軸の重なり or tolerance(30s) を見直す必要があります。")
    # 最低限、空CSVだけ出して終了
    merged.to_csv(OUTPUT_MERGED, index=False)
    print(f"⚠ 空の結合データを {OUTPUT_MERGED} に保存しました。")
    raise SystemExit(0)

# ----------------------------------------------------------
# 4. 相関解析（Δt_res vs |B|）
# ----------------------------------------------------------
if len(merged) > 2:
    corr = merged["dt_res"].corr(merged["B_total"])
    print(f"\n📈 Δt_res と |B| の相関係数: {corr:.4e}")

    # 線形フィット
    coef = np.polyfit(merged["B_total"], merged["dt_res"], 1)
    fit_line = np.poly1d(coef)
    print(f"線形回帰: Δt_res ≈ {coef[0]:.3e} × |B| + {coef[1]:.3e}")
else:
    print("⚠ データ点が少なすぎて相関解析を実行できません。")
    corr = np.nan
    coef = [0.0, 0.0]
    fit_line = lambda x: np.zeros_like(x)

# ----------------------------------------------------------
# 5. 散布図（Δt_res vs |B|）
# ----------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(merged["B_total"], merged["dt_res"], s=12, alpha=0.7, label="Data")

# フィット線を描ける場合のみプロット
if not np.isnan(corr):
    x_sorted = np.sort(merged["B_total"].values)
    plt.plot(x_sorted,
             fit_line(x_sorted),
             lw=2,
             label="Linear fit")

plt.xlabel("|B| [nT]")
plt.ylabel("Δt_res [s]")
plt.title("TFGR residuals vs Magnetic Field Strength")
plt.legend()
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_SCATTER, dpi=200)
plt.close()
print(f"✅ 散布図を保存: {OUTPUT_SCATTER}")

# ----------------------------------------------------------
# 6. 時系列プロット（Δt_res と |B|）
# ----------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = "tab:red"
ax1.set_xlabel("Time")
ax1.set_ylabel("Δt_res [s]", color=color1)
ax1.plot(merged["time"], merged["dt_res"], color=color1, lw=1.5, label="Δt_res")
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:blue"
ax2.set_ylabel("|B| [nT]", color=color2)
ax2.plot(merged["time"], merged["B_total"], color=color2, lw=1, label="|B|")
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Time Evolution: Δt_res and |B|")
fig.tight_layout()
plt.savefig(OUTPUT_TIMESERIES, dpi=200)
plt.close()
print(f"✅ 時系列図を保存: {OUTPUT_TIMESERIES}")

# ----------------------------------------------------------
# 7. CSV出力
# ----------------------------------------------------------
merged.to_csv(OUTPUT_MERGED, index=False)
print(f"✅ 結合データ保存: {OUTPUT_MERGED}")

# ----------------------------------------------------------
# 8. 結果まとめ
# ----------------------------------------------------------
print("\n=== 解析完了 ===")
print(f"📊 相関係数 r = {corr:.3e}")
print("📁 出力ファイル:")
print(f"   ├─ {OUTPUT_MERGED}")
print(f"   ├─ {OUTPUT_SCATTER}")
print(f"   └─ {OUTPUT_TIMESERIES}")
print("\n🎯 次ステップ: r の符号と大きさから、時間場–磁場結合の有無を評価します。")
