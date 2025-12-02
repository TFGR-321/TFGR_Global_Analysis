import numpy as np
import math
import matplotlib.pyplot as plt

# =====================================
# 1. TFGR parameters
# =====================================
Delta_t0 = 1.0
Lc       = 4.0e9
p        = 0.2
q        = 1.3

def Delta_t(L):
    return Delta_t0 * (1.0 + (L/Lc)**p)**q

def dDelta_t_dL(L):
    ratio = L/Lc
    return (Delta_t0*q*p/Lc *
            ratio**(p-1) * (1+ratio**p)**(q-1))

L0 = 1.0
dDt_dL_L0 = dDelta_t_dL(L0)

# =====================================
# 2. Young の実験パラメータ（光子）
# =====================================
lam = 633e-9        # 633 nm
d   = 0.25e-3       # slit separation
L0  = 1.0           # screen distance

# fringe spacing & 波数（教科書）
delta_x = L0 * lam / d
k_std   = 2*np.pi*d/(lam*L0)

print("Young fringe spacing Δx =", delta_x*1e3, "mm")

# =====================================
# 3. TFGR から同じ fringe spacing を再現
# =====================================
def omega_from_fringe(d, L0, delta_x, dDt_dL):
    return 2*np.pi*L0/(delta_x * d * dDt_dL)

omega_TF = omega_from_fringe(d, L0, delta_x, dDt_dL_L0)
k_TF     = omega_TF * dDt_dL_L0 * d / L0
lam_TF   = 2*np.pi/k_TF

print("k_std =", k_std, "  k_TF =", k_TF)
print("λ_TF (screen spacing) =", lam_TF*1e3, "mm")

# =====================================
# 4. 共通の包絡関数（ここがポイント！）
# =====================================
a_eff = 0.05e-3   # effective slit width

def envelope(x):
    # ★ λ_photon を使用：Young も TFGR も同じ包絡
    arg_env = np.pi * a_eff * x / (lam * L0)
    return np.sinc(arg_env/np.pi)**2

def young_intensity(x):
    return envelope(x) * (1 + np.cos(k_std * x))

def tfgr_intensity_with_phase(x, phi):
    # 包絡は Young と同じ、位相だけ違う
    return envelope(x) * (1 + np.cos(k_TF * x + phi))

# =====================================
# 5. φ を自動フィット
# =====================================
xs = np.linspace(-0.01, 0.01, 3000)
I_y = young_intensity(xs)
I_y /= np.max(I_y)

phi_candidates = np.linspace(-np.pi, np.pi, 2000)
errs = []

for phi in phi_candidates:
    I_tf = tfgr_intensity_with_phase(xs, phi)
    I_tf /= np.max(I_tf)
    errs.append(np.sum((I_tf - I_y)**2))

phi_best = phi_candidates[np.argmin(errs)]
print("Best phase φ =", phi_best)

# =====================================
# 6. 図を描画
# =====================================
x = np.linspace(-0.012, 0.012, 8000)
Iy  = young_intensity(x); Iy  /= np.max(Iy)
Itf = tfgr_intensity_with_phase(x, phi_best); Itf /= np.max(Itf)

plt.figure(figsize=(12,5))
plt.plot(x*1e3, Iy,  label="Standard Young (λ = 633 nm)")
plt.plot(x*1e3, Itf, "--",
         label=f"TFGR local expansion (phase-corrected φ={phi_best:.3f})")

plt.xlabel("x on screen (mm)")
plt.ylabel("normalized intensity")
plt.title("Photon double-slit: Young vs TFGR (perfect phase + envelope match)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
