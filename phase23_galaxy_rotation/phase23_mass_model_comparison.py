# phase23_mass_model_comparison.py
# TFGR 質量 vs ΛCDM ハロー質量のモデル比較 (AIC/BIC)

import numpy as np
import pandas as pd

# ==============================
# 1. データ読み込み
# ==============================
csv_file = "tfgr_wp50_clean_expanded.csv"  # 既に使っているクリーンな CSV
df = pd.read_csv(csv_file)

# 期待する列名:
# Galaxy, V0, rc, n, f_disk, RMS, chi2_red, WP50,
# logM_NFW, logM_Ein, logM_DC14   （※logM_NFV になっていたら直してください）

# 列名の typo 対策（logM_NFV → logM_NFW に自動修正）
if "logM_NFV" in df.columns and "logM_NFW" not in df.columns:
    df = df.rename(columns={"logM_NFV": "logM_NFW"})

required_cols = ["Galaxy", "V0", "logM_NFW", "logM_Ein", "logM_DC14"]
for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"必要な列 {col} が {csv_file} にありません。列名を確認してください。")

# ==============================
# 2. TFGR 質量の計算 (BTFR から)
# ==============================
# 以前の BTFR フィット結果：
# log10(M_b/Msun) = a * log10(V0/km s^-1) + b
a_btfr = 3.119
b_btfr = 3.799

logV0 = np.log10(df["V0"].values)
logM_tfgr = a_btfr * logV0 + b_btfr
df["logM_TFGR"] = logM_tfgr

# ==============================
# 3. AIC / BIC を計算する関数
# ==============================

def gaussian_loglik(residuals):
    """
    残差が正規分布 N(0, sigma^2) に従うと仮定したときの
    全データに対する対数尤度 log L を計算する。
    sigma は残差から推定。
    """
    n = len(residuals)
    # 自由度を n として単純に sigma を推定（モデル比較が主目的なのでここは簡略版）
    sigma2 = np.sum(residuals**2) / n
    sigma2 = max(sigma2, 1e-12)  # 数値安定化
    logL = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals**2 / sigma2)
    return logL


def model_aic_bic(y_obs, y_model, k):
    """
    観測値 y_obs とモデル値 y_model から
    AIC と BIC を計算する。
    k: モデルの自由パラメータ数
    """
    residuals = y_obs - y_model
    n = len(y_obs)
    logL = gaussian_loglik(residuals)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(n) - 2 * logL
    return AIC, BIC, logL


def compare_for_halo_model(name, x, y):
    """
    あるハロー質量モデル (NFW / Ein / DC14) について、
    1:1 ライン vs 直線フィットの AIC/BIC を比較。
    """
    # 有効データだけ（NaN を削除）
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    N = len(x)
    print(f"\n=== {name} モデル: 有効データ数 N = {N} ===")

    # ---- モデルA: 1:1 ライン (y = x) ----
    y_model_identity = x.copy()
    k_identity = 0  # パラメータをデータからフィットしていないので 0 とみなす
    AIC_id, BIC_id, logL_id = model_aic_bic(y, y_model_identity, k_identity)

    # ---- モデルB: 自由直線 y = a x + b ----
    # polyfit で傾きと切片をフィット
    coef, cov = np.polyfit(x, y, 1, cov=True)
    a_fit, b_fit = coef
    y_model_lin = a_fit * x + b_fit
    k_lin = 2  # パラメータ a, b の2つ
    AIC_lin, BIC_lin, logL_lin = model_aic_bic(y, y_model_lin, k_lin)

    # 結果表示
    print(f"  1:1 ライン (y = x):")
    print(f"    logL = {logL_id:8.3f},  AIC = {AIC_id:8.3f},  BIC = {BIC_id:8.3f}")
    print(f"  直線フィット (y = a x + b): a = {a_fit:.3f}, b = {b_fit:.2f}")
    print(f"    logL = {logL_lin:8.3f},  AIC = {AIC_lin:8.3f},  BIC = {BIC_lin:8.3f}")

    dAIC = AIC_id - AIC_lin
    dBIC = BIC_id - BIC_lin
    print(f"  → ΔAIC = AIC(1:1) - AIC(linear) = {dAIC:.3f}")
    print(f"  → ΔBIC = BIC(1:1) - BIC(linear) = {dBIC:.3f}")

    if dAIC > 10 and dBIC > 10:
        print("    ⇒ AIC/BIC の両方で **線形モデル (TFGR スケーリング) が圧倒的に有利**")
    elif dAIC > 4 and dBIC > 4:
        print("    ⇒ 線形モデルが**有意に優位**（中程度〜強い証拠）")
    elif dAIC > 2 or dBIC > 2:
        print("    ⇒ 線形モデルが**やや優位**")
    else:
        print("    ⇒ 1:1 ラインとの差はあまり強くない／区別がつかない")

# ==============================
# 4. 各ハロー質量モデルごとに比較
# ==============================

x_nfw = df["logM_NFW"].values
x_ein = df["logM_Ein"].values
x_dc14 = df["logM_DC14"].values
y_tfgr = df["logM_TFGR"].values

print("\n########################################")
print("#  TFGR 質量 vs ΛCDM ハロー質量  AIC/BIC 比較")
print("########################################")

compare_for_halo_model("NFW",   x_nfw,  y_tfgr)
compare_for_halo_model("Ein",   x_ein,  y_tfgr)
compare_for_halo_model("DC14",  x_dc14, y_tfgr)

print("\n※ ΔAIC, ΔBIC が正で大きいほど、TFGR の線形スケーリングの方が 1:1 ラインよりもデータに適合していることを示します。")
