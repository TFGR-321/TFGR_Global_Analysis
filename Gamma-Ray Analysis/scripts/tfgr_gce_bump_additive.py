import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# 1. GCE 観測データの読み込み
# =========================================

df = pd.read_csv("gce_40x40.csv")  # さっきの CSV ファイル名
E_data = df["E_GeV"].values
F_data = df["E2dNdE_rel"].values

logE_data = np.log(E_data)


# =========================================
# 2. ベース関数 & バンプ関数
# =========================================

def base_shape(E, p, Ec):
    """通常の IC 的なベーススペクトル E^-p * exp(-E/Ec) の形だけ"""
    return E ** (-p) * np.exp(-E / Ec)


def bump_gauss(logE, logEb, sigma):
    """logE 空間でのガウス（TFGR バンプの形）"""
    return np.exp(- (logE - logEb) ** 2 / (2.0 * sigma ** 2))


def model_additive(E, A_base, A_bump, p, Ec, Eb, sigma):
    """
    加算形モデル：
      F(E) = A_base * base(E) + A_bump * base(E) * G(E)
    """
    base = base_shape(E, p, Ec)
    G = bump_gauss(np.log(E), np.log(Eb), sigma)
    return A_base * base + A_bump * base * G


# =========================================
# 3. フィット：Eb, sigma を scan しつつ A_base, A_bump を線形解
# =========================================

# ベースの傾き・カットオフ（必要に応じて変更可）
p_fix  = 2.3
Ec_fix = 20.0  # GeV

# バンプ中心と幅のグリッド
Eb_grid    = np.logspace(np.log10(1.0), np.log10(5.0), 15)  # 1〜5 GeV
sigma_grid = np.linspace(0.2, 0.8, 13)                      # logE 幅

best_chi2  = np.inf
best_pars  = None
best_model = None

for Eb in Eb_grid:
    logEb = np.log(Eb)
    for sigma in sigma_grid:
        # 形だけ計算
        base = base_shape(E_data, p_fix, Ec_fix)
        G    = bump_gauss(logE_data, logEb, sigma)

        # デザイン行列 M : [ base, base*G ]
        X1 = base
        X2 = base * G
        M  = np.vstack([X1, X2]).T

        # 最小二乗解 [A_base, A_bump]
        coeff, _, _, _ = np.linalg.lstsq(M, F_data, rcond=None)
        A_base, A_bump = coeff

        # 物理的に A_base, A_bump <= 0 は嫌なのでペナルティ
        if (A_base <= 0) or (A_bump <= 0):
            # 大きな χ² を与えてスキップ扱い
            chi2 = 1e30
        else:
            F_model = model_additive(E_data, A_base, A_bump,
                                     p_fix, Ec_fix, Eb, sigma)
            chi2 = np.mean((F_data - F_model) ** 2)

        if chi2 < best_chi2:
            best_chi2  = chi2
            best_pars  = (A_base, A_bump, p_fix, Ec_fix, Eb, sigma)
            best_model = F_model

# =========================================
# 4. 結果表示
# =========================================

A_base_b, A_bump_b, p_b, Ec_b, Eb_b, sig_b = best_pars

print("=== TFGR additive bump fit to GCE (40x40 deg) ===")
print(f"A_base = {A_base_b:.3e}")
print(f"A_bump = {A_bump_b:.3e}")
print(f"p      = {p_b:.3f} (fixed)")
print(f"Ec     = {Ec_b:.2f} GeV (fixed)")
print(f"Eb     = {Eb_b:.2f} GeV")
print(f"sigma  = {sig_b:.3f}")
print(f"chi2   = {best_chi2:.3e}")
print(f"bump/base (peak amp ratio) ≈ {A_bump_b / A_base_b:.2e}")

# =========================================
# 5. プロット
# =========================================

# スペクトル
plt.figure(figsize=(8,6))
plt.loglog(E_data, F_data, "o", label="GCE Observed")
plt.loglog(E_data, best_model, "-", label="TFGR additive bump model")

# ベース成分とバンプ成分も分けて描くと見やすい
base_best = base_shape(E_data, p_b, Ec_b)
G_best    = bump_gauss(logE_data, np.log(Eb_b), sig_b)
plt.loglog(E_data, A_base_b*base_best, "--", label="Base only")
plt.loglog(E_data, A_bump_b*base_best*G_best, "--", label="TFGR bump only")

plt.xlabel("Eγ [GeV]")
plt.ylabel("E^2 dN/dE (relative)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("tfgr_gce_additive_fit.png")
plt.show()

# 残差
plt.figure(figsize=(8,4))
plt.semilogx(E_data, F_data - best_model, "o-")
plt.axhline(0, color="k", lw=1)
plt.xlabel("Eγ [GeV]")
plt.ylabel("Residual (data - model)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("tfgr_gce_additive_residuals.png")
plt.show()
