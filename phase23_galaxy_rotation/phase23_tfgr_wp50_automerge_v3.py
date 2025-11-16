import pandas as pd
import re

# 入力ファイル
tfgr_file = "tfgr_batch_fit_results.csv"
wp50_file = "WP50_M200.mrt.txt"

# --- TFGR 側を読み込み ---
df_tfgr = pd.read_csv(tfgr_file)
df_tfgr["Galaxy"] = df_tfgr["Galaxy"].str.strip().str.upper()
print(f"📘 Loaded TFGR: {len(df_tfgr)} galaxies")

# --- WP50 側をパース ---
rows = []
with open(wp50_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) >= 8:
            name = parts[0].upper()
            try:
                logM_NFW = float(parts[3])
                logM_Ein  = float(parts[5])
                logM_DC14 = float(parts[7])
                rows.append([name, logM_NFW, logM_Ein, logM_DC14])
            except ValueError:
                continue

df_wp = pd.DataFrame(rows, columns=["Galaxy", "logM_NFW", "logM_Ein", "logM_DC14"])
print(f"📗 Loaded WP50 halo masses: {len(df_wp)} entries")

# --- 銀河名でマージ ---
df_merge = pd.merge(df_tfgr, df_wp, on="Galaxy", how="left", suffixes=("_TFGR", "_ΛCDM"))

# --- 保存 ---
out_file = "tfgr_wp50_complete.csv"
df_merge.to_csv(out_file, index=False)

print(f"✅ 統合完了 → {out_file}")

# --- 確認表示（自動的に関連列を抽出） ---
cols = [c for c in df_merge.columns if "logM" in c]
print("📊 Columns found:", cols)
print(df_merge[["Galaxy"] + cols].head(10))
