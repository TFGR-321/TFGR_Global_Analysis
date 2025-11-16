import pandas as pd
import numpy as np
import re

# ======================================================
# Phase 23 — TFGR vs ΛCDM Halo Mass Comparison (v5)
#   NASA MRT 完全対応・不揃い行自動補正版
# ======================================================

tfgr_file = "tfgr_batch_fit_results.csv"
wp50_file = "WP50_M200.mrt.txt"
out_file  = "tfgr_wp50_merged.csv"

# --- TFGR 結果読み込み ---
print(f"Reading TFGR file: {tfgr_file}")
df_tfgr = pd.read_csv(tfgr_file)
df_tfgr.columns = [c.strip() for c in df_tfgr.columns]
df_tfgr["Galaxy"] = df_tfgr["name"].str.strip().str.upper()
print(f"  → Loaded {len(df_tfgr)} TFGR entries")

# --- Step 1: 実データ部分の抽出 ---
print(f"Scanning data start in MRT file: {wp50_file}")
data_lines = []
start_data = False
pattern = re.compile(r"^[A-Z0-9]")  # 先頭が英数字の行（NGC, UGC, DDOなど）

with open(wp50_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if pattern.match(line.strip().split()[0]):
            start_data = True
        if start_data:
            data_lines.append(line.strip())

# 空白行を除去
data_lines = [l for l in data_lines if l]

print(f"  → Extracted {len(data_lines)} data lines")

# 一時ファイルとして整形保存（pandas用）
tmp_file = "wp50_data_clean.txt"
with open(tmp_file, "w", encoding="utf-8") as f:
    for line in data_lines:
        # スペースが複数続くところを1個に圧縮
        f.write(re.sub(r"\s+", " ", line) + "\n")

# --- Step 2: pandasで再読込（不揃い行スキップ） ---
df_wp = pd.read_csv(
    tmp_file,
    sep=" ",
    names=[
        "Name", "WP50", "e_WP50",
        "logM_NFW", "e_logM_NFW",
        "logM_Ein", "e_logM_Ein",
        "logM_DC14", "e_logM_DC14"
    ],
    on_bad_lines="skip",
    engine="python"
)

print(f"  → Loaded {len(df_wp)} halo entries after cleaning")
print(f"Columns: {list(df_wp.columns)}")

# --- 銀河名統一 ---
df_wp["Galaxy"] = df_wp["Name"].astype(str).str.strip().str.upper()

# --- 不要列削除 ---
keep_cols = ["Galaxy", "WP50", "logM_NFW", "logM_Ein", "logM_DC14"]
df_wp = df_wp[keep_cols]

# --- マージ ---
print("Merging TFGR ↔ ΛCDM data...")
df_merge = pd.merge(
    df_tfgr,
    df_wp,
    on="Galaxy",
    how="outer",
    suffixes=("_TFGR", "_LCDM")
)

matched = df_merge["V0"].notna() & df_merge["WP50"].notna()
print(f"  ✅ Matched galaxies: {matched.sum()} / {len(df_merge)}")

# --- 列順整形 ---
cols_order = [
    "Galaxy", "V0", "rc", "n", "f_disk", "RMS", "chi2_red",
    "WP50", "logM_NFW", "logM_Ein", "logM_DC14"
]
for c in cols_order:
    if c not in df_merge.columns:
        df_merge[c] = np.nan
df_merge = df_merge[cols_order]

# --- 保存 ---
df_merge.to_csv(out_file, index=False)
print(f"\n✅ 統合完了: {out_file}")
print(df_merge.head(10))
