# tfgr_kappa_lambda_plot.py
# TFGR: kappa–lambda scaling (photon / electron / neutron)

import numpy as np
import matplotlib.pyplot as plt

# --- 1. データ（これまでの分析で使った値） ---
# 波長 [m]
lambda_photon  = 633e-9      # 633 nm
lambda_electron = 5.485e-12  # 50 keV 電子の de Broglie 波長 ~ 5.48 pm
lambda_neutron  = 0.18e-9    # 熱中性子 ~0.18 nm

lambdas = np.array([lambda_photon, lambda_electron, lambda_neutron])

# 対応する κ（前の 3スケールスクリプトで出た値を使用）
kappa_photon   = 5.041e8
kappa_electron = 5.817e13
kappa_neutron  = 1.773e12

kappas = np.array([kappa_photon, kappa_electron, kappa_neutron])

labels = ["Photon (633 nm)",
          "Electron (50 keV)",
          "Neutron (thermal)"]

# --- 2. κ ∝ 1/λ のフィット線を作る ---
# 単純に A = 平均(κ * λ) として κ_fit(λ) = A / λ を描く
A = np.mean(kappas * lambdas)   # ほぼ一定になっているはず

# プロット用 λ 範囲（3点をカバーするよう広めに）
lam_min = lambdas.min() / 5
lam_max = lambdas.max() * 5
lam_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), 200)

kappa_fit = A / lam_grid

print("=== kappa * lambda (各スケール) ===")
for lab, lam, kap in zip(labels, lambdas, kappas):
    print(f"{lab:23s}: κλ ≈ {kap*lam:.3e}")

print("\n平均 A = <κλ> ≈ {:.3e}".format(A))
print("理論式 κ(λ) = A / λ をプロットします。")

# --- 3. プロット（log–log） ---
plt.figure(figsize=(7, 5))

# 1/λ フィット直線
plt.plot(lam_grid, kappa_fit, label=r"fit: $\kappa \propto 1/\lambda$")

# 各スケールの点
for lam, kap, lab in zip(lambdas, kappas, labels):
    plt.scatter(lam, kap)
    plt.text(lam*1.1, kap*1.1, lab, fontsize=9)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("wavelength  λ  [m]")
plt.ylabel("TFGR coupling  κ  [arb. units]")
plt.title("TFGR κ–λ scaling (photon / electron / neutron)")

plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()
