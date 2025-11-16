"""
phase19_tfgr_fit_Vphi_model.py

phase19_tfgr_potential_ngc3198.csv に保存された
(Phi_over_c2, V_of_Phi) データを用いて、
plateau 型ポテンシャル
    V(Φ) = V0 [ 1 - exp(-(Φ/Φ_*)^m) ]
を最小二乗でフィットするスクリプト。

同じフォルダに CSV ファイルを置いて実行：
    python phase19_tfgr_fit_Vphi_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. データ読み込み
# -----------------------------
df = pd.read_csv(
    "phase19_tfgr_potential_ngc3198.csv",
    comment="#",
    names=["Phi_over_c2", "dVdPhi", "V_of_Phi"]
)
Phi = df["Phi_over_c2"].values
Vphi = df["V_of_Phi"].values

# -----------------------------
# 2. ポテンシャルモデル定義
# -----------------------------
def V_model(Phi, V0, Phi_star, m):
    return V0 * (1.0 - np.exp(- (Phi / Phi_star)**m ))

# 初期値と範囲
p0 = [np.max(Vphi), np.median(Phi), 2.0]
bounds = ([0, 1e-9, 0.5], [np.inf, 1e-6, 8.0])

# -----------------------------
# 3. フィット実行
# -----------------------------
popt, pcov = curve_fit(V_model, Phi, Vphi, p0=p0, bounds=bounds, maxfev=20000)
V0_fit, Phi_star_fit, m_fit = popt
perr = np.sqrt(np.diag(pcov))

print("\n--- Plateau Potential Fit Results ---")
print(f"V0       = {V0_fit:.3e} ± {perr[0]:.3e}")
print(f"Phi_*    = {Phi_star_fit:.3e} ± {perr[1]:.3e}")
print(f"m        = {m_fit:.3f} ± {perr[2]:.3f}")
print("-------------------------------------")

# -----------------------------
# 4. プロット
# -----------------------------
Phi_grid = np.linspace(np.min(Phi), np.max(Phi), 200)
V_model_fit = V_model(Phi_grid, *popt)

plt.figure(figsize=(7,5))
plt.plot(Phi, Vphi, 'o', label="Numerical V(Φ) data")
plt.plot(Phi_grid, V_model_fit, '-', lw=2, label=f"Fit: m={m_fit:.2f}")
plt.xlabel(r"$\Phi_t / c^2$")
plt.ylabel(r"$V(\Phi_t)$ (arb. units)")
plt.title("Analytic Plateau Potential Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
