# tfgr_kappa_lambda_with_C60.py
# TFGR: kappa–lambda scaling with C60 fullerene added

import numpy as np
import matplotlib.pyplot as plt

# --- 1. 既存3スケール（光・電子・中性子） ---

# wavelength [m]
lambda_photon   = 633e-9       # 633 nm
lambda_electron = 5.485e-12    # 50 keV electron de Broglie wavelength (~5.48 pm)
lambda_neutron  = 0.18e-9      # thermal neutron ~0.18 nm

lambda_base = np.array([lambda_photon, lambda_electron, lambda_neutron])

# corresponding kappa (from previous 3-scale analysis)
kappa_photon   = 5.041e8
kappa_electron = 5.817e13
kappa_neutron  = 1.773e12

kappa_base = np.array([kappa_photon, kappa_electron, kappa_neutron])

labels_base = [
    "Photon (633 nm)",
    "Electron (50 keV)",
    "Neutron (thermal)"
]

# --- 2. TFGR の普遍定数 A = <κλ> を、3スケールから決める ---
A = np.mean(kappa_base * lambda_base)

print("=== base 3 scales: κλ ===")
for lab, lam, kap in zip(labels_base, lambda_base, kappa_base):
    print(f"{lab:23s}: λ = {lam:.3e} m, κ = {kap:.3e},  κλ ≈ {kap*lam:.3e}")
print(f"\nTFGR universal constant A = <κλ> ≈ {A:.3e}\n")

# --- 3. C60 フラーレンの追加（代表値 λ ≈ 2.5 pm） ---
lambda_c60 = 2.5e-12   # [m] typical de Broglie wavelength for C60 interference
kappa_c60  = A / lambda_c60  # TFGR が予言する κ

print("=== C60 prediction (TFGR) ===")
print(f"lambda_C60 ≈ {lambda_c60:.3e} m")
print(f"predicted kappa_C60 ≈ {kappa_c60:.3e}")
print(f"kappa_C60 * lambda_C60 ≈ {kappa_c60*lambda_c60:.3e}\n")

# --- 4. 4点をまとめて配列化 ---
lambdas = np.concatenate([lambda_base, [lambda_c60]])
kappas  = np.concatenate([kappa_base,  [kappa_c60]])
labels  = labels_base + ["C60 fullerene (~2.5 pm)"]

# --- 5. フィット線 κ(λ) = A/λ を描く ---
lam_min = lambdas.min() / 2.0
lam_max = lambdas.max() * 2.0
lam_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), 400)
kappa_fit = A / lam_grid

# --- 6. プロット（log-log） ---
plt.figure(figsize=(7, 5))

# 1/λ フィット直線
plt.plot(lam_grid, kappa_fit, label=r"fit: $\kappa(\lambda)=A/\lambda$")

# 4つの点をプロット
colors = ["C0", "C1", "C2", "C3"]
for lam, kap, lab, col in zip(lambdas, kappas, labels, colors):
    plt.scatter(lam, kap, color=col)
    plt.text(lam*1.1, kap*1.1, lab, fontsize=9)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("wavelength  λ  [m]")
plt.ylabel("TFGR coupling  κ  [arb. units]")
plt.title("TFGR κ–λ scaling (photon / electron / neutron / C60)")

plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()
