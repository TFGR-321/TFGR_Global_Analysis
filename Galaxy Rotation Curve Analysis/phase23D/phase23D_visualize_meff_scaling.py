import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# データ読み込み
# ===============================
meff_df = pd.read_csv("phase23C_Meff_results.csv")

# (必要に応じて) AIC/BIC結果を再計算
def calc_aic_bic(y_obs, y_model, k):
    resid = y_obs - y_model
    n = len(y_obs)
    sse = np.sum(resid**2)
    sigma2 = sse / n
    logL = -0.5 * n * (np.log(2*np.pi*sigma2) + 1)
    AIC = 2*k - 2*logL
    BIC = k*np.log(n) - 2*logL
    return AIC, BIC, logL

x = meff_df["logM_halo"].values
y = meff_df["logM_eff"].values

# 1:1モデル
AIC_1, BIC_1, logL_1 = calc_aic_bic(y, x, 1)

# 線形フィット
coef = np.polyfit(x, y, 1)
a, b = coef
y_fit = a*x + b
AIC_lin, BIC_lin, logL_lin = calc_aic_bic(y, y_fit, 2)

ΔAIC = AIC_1 - AIC_lin
ΔBIC = BIC_1 - BIC_lin

print("======================================")
print(" Phase 23-D Visualization Summary")
print("======================================")
print(f"Best-fit: y = {a:.3f}x + {b:.3f}")
print(f"ΔAIC = {ΔAIC:.3f},  ΔBIC = {ΔBIC:.3f}")
print("======================================")

# ===============================
# 図1: logM_eff vs logM_halo
# ===============================
plt.figure(figsize=(7,6))
sns.scatterplot(x=x, y=y, color='dodgerblue', s=60, label='TFGR data')
plt.plot(x, x, 'k--', label='ΛCDM: y=x (1:1)')
plt.plot(x, y_fit, 'r-', lw=2, label=f'TFGR fit (a={a:.2f}, b={b:.2f})')
plt.xlabel(r'$\log M_{\mathrm{halo}}\ (Λ\mathrm{CDM})$')
plt.ylabel(r'$\log M_{\mathrm{eff}}\ (\mathrm{TFGR})$')
plt.title("TFGR Effective Mass Scaling vs ΛCDM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("TFGR_vs_LCDM_Meff_scaling.png", dpi=300)
plt.close()

# ===============================
# 図2: ΔAIC / ΔBIC ヒストグラム
# ===============================
plt.figure(figsize=(6,5))
bars = [ΔAIC, ΔBIC]
labels = [r'$\Delta \mathrm{AIC}$', r'$\Delta \mathrm{BIC}$']
colors = ['royalblue', 'orange']
plt.bar(labels, bars, color=colors)
plt.axhline(0, color='k', lw=1)
plt.ylabel('Δ Value (ΛCDM − TFGR)')
plt.title('Model Comparison: AIC/BIC Advantage')
for i, val in enumerate(bars):
    plt.text(i, val+2, f'{val:.1f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("TFGR_AIC_BIC_comparison.png", dpi=300)
plt.close()

# ===============================
# 出力メッセージ
# ===============================
print("✅ 出力完了:")
print("   - TFGR_vs_LCDM_Meff_scaling.png")
print("   - TFGR_AIC_BIC_comparison.png")
print("これらの図で Phase 23-B の理論的傾向が視覚的に再現されます。")
