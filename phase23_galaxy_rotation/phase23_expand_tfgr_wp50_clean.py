import pandas as pd

# === 入力ファイル ===
batch_file = "tfgr_batch_fit_results.csv"   # 各銀河のTFGRパラメータ
wp50_file  = "WP50_M200.mrt.txt"            # ΛCDM側ハロー質量
out_file   = "tfgr_wp50_clean_expanded.csv"

# === ファイル読み込み ===
df_tfgr = pd.read_csv(batch_file)
df_tfgr.rename(columns={"name": "Galaxy"}, inplace=True)

# WP50 データ読み込み
df_wp = pd.read_csv(wp50_file, delim_whitespace=True, header=None,
                    names=["Galaxy", "col1", "col2", "logM_NFW", "err_NFW", "logM_Ein", "err_Ein", "logM_DC14", "err_DC14"])

# === マージ ===
merged = pd.merge(df_tfgr, df_wp[["Galaxy", "logM_NFW", "logM_Ein", "logM_DC14"]], on="Galaxy", how="inner")

# === 保存 ===
merged.to_csv(out_file, index=False)

print(f"✅ 出力完了: {out_file}")
print(f"📈 銀河数: {len(merged)} 件")
print("列:", ", ".join(merged.columns))
