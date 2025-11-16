import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 万有引力定数 [kpc·(km/s)^2 / M☉]
G = 4.30091e-6

# データ読み込み
df = pd.read_csv("tfgr_wp50_clean.csv")

# TFGRによる有効ハロー質量を計算
df["logM_TFGR"] = np.log10((df["V0"]**2 * df["rc"]) / G)

# 3種類のΛCDMモデルと比較
for model in ["NFW", "Ein", "DC14"]:
    col = f"logM_{model}"
    if col in df.columns:
        valid = df[[col, "logM_TFGR"]].dropna()
        diff = valid["logM_TFGR"] - valid[col]

        print(f"\n=== {model} モデル ===")
        print(f"平均差 ΔlogM = {diff.mean():.3f}")
        print(f"標準偏差 RMS = {diff.std():.3f}")
        print(f"相関係数 r = {valid.corr().iloc[0,1]:.3f}")

        # 散布図を描画
        plt.figure(figsize=(7,6))
        plt.scatter(valid[col], valid["logM_TFGR"], s=70, edgecolor="k", color="skyblue")
        plt.plot([9.5, 13.8], [9.5, 13.8], "r--", lw=1.3, label="1:1 line")
        plt.xlabel(f"log M_200 ({model}) [M☉]")
        plt.ylabel("log M_TFGR [M☉]")
        plt.title(f"TFGR と ΛCDM のハロー質量比較 ({model})")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"TFGR_vs_{model}_mass_clean.png", dpi=250)
        plt.close()
