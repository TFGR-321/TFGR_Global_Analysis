import pandas as pd

# 元ファイルを読み込み
df = pd.read_csv("tfgr_wp50_complete.csv")

# 不要な空列 (_x 系) を削除
df = df.drop(columns=[c for c in df.columns if c.endswith("_x")], errors="ignore")

# _y を削除してシンプルな列名に変更
df = df.rename(columns={
    "logM_NFW_y": "logM_NFW",
    "logM_Ein_y": "logM_Ein",
    "logM_DC14_y": "logM_DC14"
})

# 保存
df.to_csv("tfgr_wp50_clean.csv", index=False)

print("✅ クリーン版を出力しました → tfgr_wp50_clean.csv")
print(df.head(10)[["Galaxy", "logM_NFW", "logM_Ein", "logM_DC14"]])
