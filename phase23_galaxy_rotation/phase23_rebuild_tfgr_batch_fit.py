###############################################
# phase23_rebuild_tfgr_batch_fit.py
# Author: Mitsui Takahiro
# Purpose: Rebuild tfgr_batch_fit_results.csv
#          from individual galaxy TFGR fit CSVs
###############################################

import pandas as pd
import glob
import os

# 出力ファイル名
output_file = "tfgr_batch_fit_results.csv"

# 現在のディレクトリ内で .csv ファイルを全取得（自身のtfgr_batch_fit_results.csvは除外）
csv_files = [f for f in glob.glob("*.csv") if f != output_file]

if not csv_files:
    print("⚠️  銀河CSVファイルが見つかりません。スクリプトと同じフォルダに配置してください。")
    exit()

records = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        galaxy_name = os.path.splitext(os.path.basename(file))[0]

        # 各ファイルの列を統一的に抽出（存在しない場合はNaN扱い）
        rec = {
            "Galaxy": galaxy_name,
            "V0": df.get("V0", [None])[0],
            "rc": df.get("rc", [None])[0],
            "n": df.get("n", [None])[0],
            "f_disk": df.get("f_disk", [None])[0],
            "RMS": df.get("RMS", [None])[0],
            "chi2_red": df.get("chi2_red", [None])[0]
        }
        records.append(rec)
        print(f"✅ Loaded: {galaxy_name}")

    except Exception as e:
        print(f"⚠️  読み込み失敗: {file} ({e})")

# 統合してCSV出力
df_all = pd.DataFrame(records)
df_all = df_all.sort_values("Galaxy")
df_all.to_csv(output_file, index=False)

print("\n========================================")
print(f"📦  出力完了: {output_file}")
print(f"🪐  銀河数: {len(df_all)} 件")
print("========================================")
