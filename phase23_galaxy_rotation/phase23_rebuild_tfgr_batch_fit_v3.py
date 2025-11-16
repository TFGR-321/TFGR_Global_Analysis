# ===============================================
# phase23_rebuild_tfgr_batch_fit_v3.py
# TFGR 個別フィット結果（各銀河CSV）を統合して
# tfgr_batch_fit_results.csv を再構築する
# ===============================================

import pandas as pd
import glob

rows = []
print("🔍 Searching for *_tfgr_fit.csv files...")

# 今回は各銀河の個別CSVが「_tfgr_fit.csv」ではなく「.csv」で終わっているため、
# 惑星や付随データを除外して主要銀河だけを抽出
exclude = ["phase", "tfgr_", "wp50", "SPARC", "WP50", "list", "plots", "Figure"]

for path in sorted(glob.glob("*.csv")):
    # 除外条件
    if any(x in path for x in exclude):
        continue

    try:
        df = pd.read_csv(path)
        # 有効列だけ抽出して統計量を取る（単一行データにも対応）
        df_num = df.select_dtypes(include=["number"])
        row = df_num.mean().to_dict()
        row["Galaxy"] = path.replace(".csv", "")
        rows.append(row)
        print(f"✅ Loaded: {path}")
    except Exception as e:
        print(f"⚠️ Skipped {path}: {e}")

# DataFrame統合
df_all = pd.DataFrame(rows)

# 欠損を除外
df_all = df_all.dropna(how="all")

# 列の順序統一（存在する列だけ残す）
preferred_cols = ["Galaxy", "V0", "rc", "n", "f_disk", "RMS", "chi2_red"]
df_all = df_all[[c for c in preferred_cols if c in df_all.columns]]

# 出力
outpath = "tfgr_batch_fit_results.csv"
df_all.to_csv(outpath, index=False)

print("\n========================================")
print(f"📦 出力完了: {outpath}")
print(f"🪐 銀河数: {len(df_all)} 件")
print("========================================")
