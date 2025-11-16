import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ============================================
# TFGR vs ΛCDM 理論比較モード
# ============================================

def V_TFGR(r, V0, rc, n):
    return V0 * np.sqrt(1 - np.exp(-(r/rc)**n))

def V_NFW(r, V0, rc):
    return V0 * (r/rc) / (1 + r/rc)

def V_Ein(r, V0, rc, n):
    return V0 * (1 - np.exp(-(r/rc)**(1/n)))

def V_DC14(r, V0, rc):
    return V0 * ((r/rc) / (1 + (r/rc)**1.5))

def calc_aic_bic(y_obs, y_model, k):
    resid = y_obs - y_model
    n = len(y_obs)
    sse = np.sum(resid**2)
    sigma2 = sse / n
    logL = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(n) - 2 * logL
    return AIC, BIC, logL

# ============================================
# CSV 読み込み
# ============================================

df = pd.read_csv("tfgr_wp50_clean_expanded.csv")
results = []

for _, row in df.iterrows():
    name = row["Galaxy"] if "Galaxy" in row else f"Galaxy_{_}"

    # パラメータ取得
    V0 = row.get("V0", np.nan)
    rc = row.get("rc", np.nan)
    n = row.get("n", np.nan)
    if np.isnan(V0) or np.isnan(rc) or np.isnan(n):
        continue

    # 半径配列（0〜10×r_c）
    r = np.linspace(0.1, 10*rc, 50)
    v_true = V_TFGR(r, V0, rc, n)
    v_obs = v_true * (1 + np.random.normal(0, 0.1, len(r)))  # ±10% ノイズ

    # 各モデルフィット
    try:
        popt_tfgr, _ = curve_fit(V_TFGR, r, v_obs, p0=[V0, rc, n], maxfev=5000)
        popt_nfw, _ = curve_fit(V_NFW, r, v_obs, p0=[V0, rc], maxfev=5000)
        popt_ein, _ = curve_fit(V_Ein, r, v_obs, p0=[V0, rc, n], maxfev=5000)
        popt_dc14, _ = curve_fit(V_DC14, r, v_obs, p0=[V0, rc], maxfev=5000)
    except:
        continue

    v_fit_tfgr = V_TFGR(r, *popt_tfgr)
    v_fit_nfw = V_NFW(r, *popt_nfw)
    v_fit_ein = V_Ein(r, *popt_ein)
    v_fit_dc14 = V_DC14(r, *popt_dc14)

    AIC_tfgr, BIC_tfgr, logL_tfgr = calc_aic_bic(v_obs, v_fit_tfgr, 3)
    AIC_nfw, BIC_nfw, logL_nfw = calc_aic_bic(v_obs, v_fit_nfw, 2)
    AIC_ein, BIC_ein, logL_ein = calc_aic_bic(v_obs, v_fit_ein, 3)
    AIC_dc14, BIC_dc14, logL_dc14 = calc_aic_bic(v_obs, v_fit_dc14, 2)

    results.append({
        "Galaxy": name,
        "AIC_TFGR": AIC_tfgr, "BIC_TFGR": BIC_tfgr, "logL_TFGR": logL_tfgr,
        "AIC_NFW": AIC_nfw, "BIC_NFW": BIC_nfw, "logL_NFW": logL_nfw,
        "AIC_Ein": AIC_ein, "BIC_Ein": BIC_ein, "logL_Ein": logL_ein,
        "AIC_DC14": AIC_dc14, "BIC_DC14": BIC_dc14, "logL_DC14": logL_dc14
    })

# ============================================
# 出力
# ============================================

out = pd.DataFrame(results)
out.to_csv("phase23B_model_comparison_results.csv", index=False)

# 統計まとめ
def win_rate(col1, col2):
    return np.mean(out[col1] < out[col2]) * 100

summary = {
    "TFGR_vs_NFW_AIC_win%": win_rate("AIC_TFGR", "AIC_NFW"),
    "TFGR_vs_Ein_AIC_win%": win_rate("AIC_TFGR", "AIC_Ein"),
    "TFGR_vs_DC14_AIC_win%": win_rate("AIC_TFGR", "AIC_DC14"),
    "TFGR_vs_NFW_BIC_win%": win_rate("BIC_TFGR", "BIC_NFW"),
    "TFGR_vs_Ein_BIC_win%": win_rate("BIC_TFGR", "BIC_Ein"),
    "TFGR_vs_DC14_BIC_win%": win_rate("BIC_TFGR", "BIC_DC14"),
}

print("=======================================")
print("  TFGR vs ΛCDM モデル 理論比較結果")
print("=======================================")
print(out.describe())
print("---------------------------------------")
print("TFGR 勝率 (%)")
for k, v in summary.items():
    print(f"{k}: {v:.1f}%")
print("=======================================")
print("出力: phase23B_model_comparison_results.csv")
