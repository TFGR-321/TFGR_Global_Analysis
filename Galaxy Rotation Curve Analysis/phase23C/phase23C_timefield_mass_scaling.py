import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ----------------------------
# TFGR 有効質量関数
# ----------------------------
def M_eff(M0, L, Lc=4e9, p=0.21, q=1.32):
    return M0 * (1 + (L/Lc)**p)**q

# ----------------------------
# 統計補助関数
# ----------------------------
def calc_aic_bic(y_obs, y_model, k):
    resid = y_obs - y_model
    n = len(y_obs)
    sse = np.sum(resid**2)
    sigma2 = sse / n
    logL = -0.5 * n * (np.log(2*np.pi*sigma2) + 1)
    AIC = 2*k - 2*logL
    BIC = k*np.log(n) - 2*logL
    return AIC, BIC, logL

# ----------------------------
# データ読み込み
# ----------------------------
df = pd.read_csv("phase23B_model_comparison_results.csv")  # 参照情報（銀河名など）
dfm = pd.read_csv("tfgr_wp50_clean_expanded.csv")

# 仮のスケール距離 L を定義（半径スケール proxy）
dfm["L"] = dfm["rc"] * 3.09e19  # [m] に変換 (1 kpc ≈ 3.09e19 m)
dfm["M0"] = 10**dfm["logM_NFW"]  # NFW 質量を M0 と仮定

# ----------------------------
# TFGR有効質量を算出
# ----------------------------
dfm["M_eff"] = M_eff(dfm["M0"], dfm["L"])
dfm["logM_eff"] = np.log10(dfm["M_eff"])
dfm["logM_halo"] = dfm["logM_NFW"]

# ----------------------------
# 1:1 vs 線形モデル フィット
# ----------------------------
x = dfm["logM_halo"].values
y = dfm["logM_eff"].values

# 1:1 ライン
AIC_1to1, BIC_1to1, logL_1to1 = calc_aic_bic(y, x, 1)

# 線形フィット
def linear(x, a, b):
    return a*x + b

popt, _ = curve_fit(linear, x, y, p0=[1, 0])
y_fit = linear(x, *popt)
AIC_lin, BIC_lin, logL_lin = calc_aic_bic(y, y_fit, 2)

# ----------------------------
# 結果出力
# ----------------------------
print("==============================================")
print(" TFGR 有効質量スケーリング vs ΛCDM")
print("==============================================")
print(f"Fit result: a = {popt[0]:.3f}, b = {popt[1]:.3f}")
print(f"logL(1:1) = {logL_1to1:.3f}, AIC(1:1) = {AIC_1to1:.3f}, BIC(1:1) = {BIC_1to1:.3f}")
print(f"logL(linear) = {logL_lin:.3f}, AIC(linear) = {AIC_lin:.3f}, BIC(linear) = {BIC_lin:.3f}")
print("----------------------------------------------")
print(f"ΔAIC = {AIC_1to1 - AIC_lin:.3f}")
print(f"ΔBIC = {BIC_1to1 - BIC_lin:.3f}")
if AIC_1to1 > AIC_lin:
    print("⇒ TFGR補正付きスケーリングの方がデータに適合")
else:
    print("⇒ ΛCDM (1:1) モデルの方が適合")
print("==============================================")

# 保存
pd.DataFrame({
    "Galaxy": dfm["Galaxy"],
    "logM_halo": dfm["logM_halo"],
    "logM_eff": dfm["logM_eff"]
}).to_csv("phase23C_Meff_results.csv", index=False)
