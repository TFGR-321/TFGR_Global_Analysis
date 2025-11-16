import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ファイル読込
df = pd.read_csv("tfgr_wp50_clean.csv")

# NaNを除外
df = df.dropna(subset=["logM_NFW", "logM_Ein", "logM_DC14"])

# TFGR実効質量の推定（ここではV0を代理スケールとして簡易変換）
# 実際のTFGR質量を別で算出している場合は、その列名に置き換えてOK
df["logM_TFGR"] = np.log10(df["V0"]**2 * df["rc"]) - 1.5  # 相対比較スケール化

models = {
    "NFW": "logM_NFW",
    "Ein": "logM_Ein",
    "DC14": "logM_DC14"
}

# 結果格納
results = []

for name, col in models.items():
    x = df[col].values
    y = df["logM_TFGR"].values

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value**2

    results.append([name, slope, intercept, r2, p_value])

    # 図を作成
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=70, color="skyblue", edgecolor="k", alpha=0.7)
    plt.plot(x, slope * x + intercept, "r--", label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    plt.plot(x, x, "gray", linestyle=":", label="1:1 line")
    plt.xlabel(f"log M_200 ({name}) [M☉]")
    plt.ylabel("log M_TFGR [M☉]")
    plt.title(f"TFGR vs ΛCDM 質量対応（{name}）")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"TFGR_vs_{name}_regression.png", dpi=200)
    plt.close()

# 結果をDataFrame化
df_res = pd.DataFrame(results, columns=["Model", "Slope", "Intercept", "R²", "p_value"])

# 表示
print("📊 TFGR vs ΛCDM 回帰結果")
print(df_res.to_string(index=False))

# 保存
df_res.to_csv("phase23_mass_regression_results.csv", index=False)
print("\n✅ 結果を保存しました → phase23_mass_regression_results.csv")
