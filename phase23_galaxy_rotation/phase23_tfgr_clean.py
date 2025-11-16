import pandas as pd
import numpy as np

# 元のファイルを読み込む（あなたが作成済みの統合ファイル）
df = pd.read_csv("tfgr_wp50_merged.csv")

# 数値が入っている銀河だけを抽出
df = df[pd.to_numeric(df["V0"], errors="coerce").notna()]
df = df[df["V0"] > 0]

# 不要な文字行を削除
df = df[~df["Galaxy"].astype(str).str.contains("NONE|UNITS|HALO|TIMES|solMass|Label|file", case=False, na=False)]

# 新しいCSVとして保存
df.to_csv("tfgr_wp50_clean.csv", index=False)
print(f"✅ クリーン済みデータ {len(df)} 行を保存しました（tfgr_wp50_clean.csv）")
print(df.head())
