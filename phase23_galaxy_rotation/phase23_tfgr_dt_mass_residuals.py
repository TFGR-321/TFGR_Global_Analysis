# phase23_tfgr_dt_mass_residuals.py
# TFGR Δt(L) - ΔM 相関解析 (Phase 23-C)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -----------------------------
# 1. データ読み込み
# -----------------------------
csv_file = "tfgr_wp50_clean.csv"
df = pd.read_csv(csv_file)

# 列名確認と修正
if "logM_NFV" in df.columns and "logM_NFW" not in df.columns:
    df = df.rename(columns={"logM_NFV": "logM_NFW"})

# -----------------------------
# 2. TFGR 質量の再計算（BTFR式）
# -----------------------------
a_btfr, b_btfr = 3.119, 3.799
df["logM_TFGR"] = a_btfr * np.log10(df["V0"]) + b_btfr

# -----------------------------
# 3. TFGR - ΛCDM 質量差 ΔM を計算
# -----------------------------
df["dM_NFW"]  = df["logM_TFGR"] - df["logM_NFW"]
df["dM_Ein"]  = df["logM_TFGR"] - df["logM_Ein"]
df["dM_DC14"] = df["logM_TFGR"] - df["logM_DC14"]

# -----------------------------
# 4. TFGR補正関数 Δt(L)/Δt₀ を計算
# -----------------------------
# パラメータ（Phase 17 以降で確立済み）
Lc = 4.0e9     # 臨界長スケール [m]
p, q = 0.21, 1.32
L_values = df["rc"].values * 3.086e19  # rc[kpc] -> m 変換 (1 kpc = 3.086e19 m)
df["Delta_t_ratio"] = (1 + (L_values / Lc)**p)**q - 1  # Δt(L)/Δt₀ - 1

# -----------------------------
# 5. Δt と ΔM の相関を解析
# -----------------------------
models = [("NFW", "dM_NFW"), ("Ein", "dM_Ein"), ("DC14", "dM_DC14")]
results = []

for label, col in models:
    x = df["Delta_t_ratio"]
    y = df[col]
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r_value, p_value, stderr = linregress(x[mask], y[mask])
    results.append((label, slope, intercept, r_value**2, p_value))

    # プロット
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, color="skyblue", edgecolor="k", s=60)
    plt.plot(x, slope*x + intercept, "r--", lw=2, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
    plt.xlabel("Δt(L)/Δt₀ − 1  (scale-dependent time correction)")
    plt.ylabel(f"Δlog M (TFGR − {label}) [dex]")
    plt.title(f"TFGR補正 Δt(L) vs 質量残差 ΔM ({label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"TFGR_dt_masscorr_{label}.png", dpi=200)
    plt.close()

# -----------------------------
# 6. 結果をCSV出力
# -----------------------------
res_df = pd.DataFrame(results, columns=["Model", "Slope", "Intercept", "R2", "p_value"])
res_df.to_csv("phase23_dt_masscorr_summary.csv", index=False)

print("=== Δt(L) vs ΔM 相関解析完了 ===")
print(res_df)
print("\nR² が高く p < 0.05 のモデルでは、TFGR補正 Δt(L) が ΛCDMとの質量差を統計的に説明しています。")
print("出力画像: TFGR_dt_masscorr_NFW.png, TFGR_dt_masscorr_Ein.png, TFGR_dt_masscorr_DC14.png")
print("出力表: phase23_dt_masscorr_summary.csv")
