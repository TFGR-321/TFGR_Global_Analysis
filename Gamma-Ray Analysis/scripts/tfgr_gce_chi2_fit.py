# tfgr_gce_chi2_fit.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. GCE データ（Daylan+ 40x40deg 図から抽出した相対スペクトル）
# -----------------------------
E_GeV = np.array([
    0.313850125,
    0.403188185,
    0.501307123,
    0.637029526,
    0.800728792,
    0.984808344,
    1.265135989,
    1.607655152,
    2.020778811,
    2.512550839,
    3.192791398,
    3.969781682,
    5.044548596,
    6.410294714,
    7.970289118,
    10.01843669,
    12.73079851,
    16.17749713,
    20.11441518,
    25.28327293,
    32.12839122,
    39.94708174
])

F_rel = np.array([
    5.25E-07,
    8.95E-07,
    5.06E-06,
    5.53E-06,
    4.18E-05,
    2.62E-05,
    9.09E-05,
    3.82E-05,
    6.96E-05,
    3.58E-05,
    2.40E-05,
    5.10E-05,
    7.06E-06,
    6.41E-07,
    3.88E-06,
    1.70E-06,
    8.95E-07,
    3.85E-07,
    1.99E-06,
    1.07E-06,
    1.08E-07,
    4.70E-07
])

# -----------------------------
# 2. 簡易エラーモデル（相対誤差）
#    Fig.6 のエラーバーを見ての近似：
#    E<1GeV: 30%, 1-5GeV:20%, 5-20GeV:30%, >20GeV:40%
# -----------------------------
def relative_error(E):
    if E < 1.0:
        return 0.30
    elif E < 5.0:
        return 0.20
    elif E < 20.0:
        return 0.30
    else:
        return 0.40

rel_err = np.array([relative_error(E) for E in E_GeV])
sigma = rel_err * F_rel

# 念のため、ゼロ除けの下限を少し入れておく（理論的には不要）
sigma_floor = 1e-3 * np.max(F_rel)
sigma = np.maximum(sigma, sigma_floor)

# -----------------------------
# 3. モデル定義
#    ベース：A_base * E^{-p} * exp(-E/Ec)
#    TFGRバンプ：上に log-Gauss を掛けた成分を加算
# -----------------------------
p_fixed = 2.3      # 電子スペクトルの傾き（固定）
Ec_fixed = 20.0    # 高エネルギーカットオフ GeV（固定）

def base_model(E, A_base):
    """切断付きパワー則だけのベースモデル"""
    return A_base * E**(-p_fixed) * np.exp(-E / Ec_fixed)

def tfgr_additive_model(E, A_base, A_bump, Eb, sigma_b):
    """
    ベース + TFGR 由来の log-Gaussian バンプ
    F(E) = A_base * E^{-p} e^{-E/Ec}
         + A_bump * E^{-p} e^{-E/Ec} * exp[-(ln(E/Eb))^2 / (2 sigma_b^2)]
    """
    base = base_model(E, A_base)
    log_term = np.log(E / Eb)
    bump_shape = np.exp(-0.5 * (log_term / sigma_b)**2)
    bump = A_bump * E**(-p_fixed) * np.exp(-E / Ec_fixed) * bump_shape
    return base + bump

# -----------------------------
# 4. ベースモデルのフィット（比較用）
# -----------------------------
p0_base = [1e-9]  # 初期値
bounds_base = ([0.0], [np.inf])

popt_base, pcov_base = curve_fit(
    base_model, E_GeV, F_rel,
    p0=p0_base,
    sigma=sigma,
    absolute_sigma=True,
    bounds=bounds_base
)

A_base_best = popt_base[0]
F_base_best = base_model(E_GeV, A_base_best)

# chi2, AIC, BIC 計算
def compute_stats(y_obs, y_model, y_err, n_params):
    chi2 = np.sum(((y_obs - y_model) / y_err)**2)
    N = len(y_obs)
    k = n_params
    dof = N - k
    AIC = chi2 + 2*k
    BIC = chi2 + k * np.log(N)
    return chi2, dof, AIC, BIC

chi2_base, dof_base, AIC_base, BIC_base = compute_stats(
    F_rel, F_base_best, sigma, n_params=1
)

# -----------------------------
# 5. TFGR 加算バンプモデルのフィット
# -----------------------------
# 以前の解析結果を元にしたそこそこ良い初期値
p0_tfgr = [
    1e-9,      # A_base
    5e-4,      # A_bump
    3.5,       # Eb (GeV)
    0.55       # sigma_b
]

# パラメータの範囲
# A_base, A_bump > 0, Eb は 0.3〜40 GeV, sigma_b は 0.1〜2
lower_bounds = [0.0, 0.0, 0.3, 0.1]
upper_bounds = [np.inf, np.inf, 40.0, 2.0]

popt_tfgr, pcov_tfgr = curve_fit(
    tfgr_additive_model, E_GeV, F_rel,
    p0=p0_tfgr,
    sigma=sigma,
    absolute_sigma=True,
    bounds=(lower_bounds, upper_bounds)
)

A_base_fit, A_bump_fit, Eb_fit, sigma_b_fit = popt_tfgr
F_tfgr_best = tfgr_additive_model(E_GeV, *popt_tfgr)

chi2_tfgr, dof_tfgr, AIC_tfgr, BIC_tfgr = compute_stats(
    F_rel, F_tfgr_best, sigma, n_params=4
)

# -----------------------------
# 6. 結果表示
# -----------------------------
print("=== Base model only (cutoff power-law) ===")
print(f" A_base = {A_base_best:.3e}")
print(f" chi2   = {chi2_base:.3e}")
print(f" dof    = {dof_base}")
print(f" AIC    = {AIC_base:.3e}")
print(f" BIC    = {BIC_base:.3e}")
print()

print("=== TFGR additive bump model ===")
print(f" A_base = {A_base_fit:.3e}")
print(f" A_bump = {A_bump_fit:.3e}")
print(f" Eb     = {Eb_fit:.2f} GeV")
print(f" sigma_b= {sigma_b_fit:.3f}")
print(f" chi2   = {chi2_tfgr:.3e}")
print(f" dof    = {dof_tfgr}")
print(f" AIC    = {AIC_tfgr:.3e}")
print(f" BIC    = {BIC_tfgr:.3e}")
print()

print("=== Model comparison (小さいほど良い) ===")
print(f" ΔAIC (TFGR - Base) = {AIC_tfgr - AIC_base:.3e}")
print(f" ΔBIC (TFGR - Base) = {BIC_tfgr - BIC_base:.3e}")

# -----------------------------
# 7. プロット（スペクトル ＋ 残差）
# -----------------------------
fig1, ax1 = plt.subplots(figsize=(7, 5))

ax1.errorbar(E_GeV, F_rel, yerr=sigma, fmt='o', color='C0',
             label='GCE data (40x40°)')
E_plot = np.logspace(np.log10(min(E_GeV)*0.8),
                     np.log10(max(E_GeV)*1.2), 400)

ax1.plot(E_plot,
         base_model(E_plot, A_base_best),
         '--', color='C2', label='Base only (fit)')
ax1.plot(E_plot,
         tfgr_additive_model(E_plot, *popt_tfgr),
         '-', color='C1', label='TFGR additive bump (fit)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("Eγ [GeV]")
ax1.set_ylabel("E² dN/dE (relative units)")
ax1.grid(True, which='both', ls=':', alpha=0.4)
ax1.legend()
fig1.tight_layout()
fig1.savefig("gce_tfgr_chi2_fit_spectrum.png", dpi=200)

# 残差（data - model）を TFGR モデルに対して
fig2, ax2 = plt.subplots(figsize=(7, 3))

residual = F_rel - F_tfgr_best
ax2.errorbar(E_GeV, residual, yerr=sigma, fmt='o', color='C0')
ax2.axhline(0.0, color='k', lw=1)

ax2.set_xscale('log')
ax2.set_xlabel("Eγ [GeV]")
ax2.set_ylabel("Residual (data - model)")
ax2.grid(True, which='both', ls=':', alpha=0.4)
fig2.tight_layout()
fig2.savefig("gce_tfgr_chi2_fit_residuals.png", dpi=200)

plt.show()
