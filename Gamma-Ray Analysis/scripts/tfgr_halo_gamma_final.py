import numpy as np
import matplotlib.pyplot as plt

# ============================
# TFGR PARAMETERS
# ============================
Lc = 4.0e9          # [m]
p  = 0.19
q  = 1.29
c  = 3.0e8          # [m/s]

def delta_t(L):
    return (1 + (L / Lc)**p)**q

def a_t(L):
    x = L / Lc
    return -c**2 * q * p * x**(p-1) / (Lc * (1 + x**p)**(1-q))

def A_coeff(L):
    return a_t(L) / c  # [1/s]

# ============================
# ELECTRON MODEL
# ============================

E_min = 1e-2   # GeV (10 MeV)
E_max = 1e4    # GeV (10 TeV)
n_E   = 400
E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_E)

Q0 = 1.0
s_inj = 2.2

def Q_inj(E):
    return Q0 * E**(-s_inj)

# 損失係数：以前ピークがきれいに出ていた値
# --- energy loss coefficient (tuned so that Ecross ~ 40 GeV at 20 kpc) ---
b0 = 1.65e-12  # [1/s/GeV]


def E_dot(E, A_L):
    """エネルギー変化率 dE/dt"""
    # A_L (<0) を加速項として使うので符号を反転
    return (-A_L) * E - b0 * E**2

def steady_spectrum(E_grid, A_L):
    """
    定常状態 n_e(E) ≈ 1/|dE/dt| ∫_E^{Emax} Q(E') dE'
    """
    Q_vals = Q_inj(E_grid)
    E_rev = E_grid[::-1]
    Q_rev = Q_vals[::-1]
    integ_rev = np.cumsum(Q_rev[:-1] * np.diff(E_rev))
    integ = np.zeros_like(E_grid)
    integ[:-1] = integ_rev[::-1]
    integ[-1] = 0.0

    dEdt = E_dot(E_grid, A_L)
    mask = np.abs(dEdt) > 1e-40
    nE = np.zeros_like(E_grid)
    nE[mask] = integ[mask] / np.abs(dEdt[mask])
    return nE

# 半径リスト：内側〜ハロー外縁まで
kpc = 3.086e19
radii_kpc = [20.0, 50.0, 100.0]
radii_m   = [r * kpc for r in radii_kpc]

# 基準（TFGRなし）
A_zero = 0.0
nE_no  = steady_spectrum(E_grid, A_zero)

# 各半径で電子スペクトルを計算
nE_tfgr = {}
A_vals  = {}
for r_kpc, r_m in zip(radii_kpc, radii_m):
    A_L = A_coeff(r_m)
    A_vals[r_kpc] = A_L
    nE_tfgr[r_kpc] = steady_spectrum(E_grid, A_L)
    print(f"[electrons] R={r_kpc:5.1f} kpc: A(L)={A_L:.3e} 1/s, "
          f"E_cross≈{abs(A_L)/b0:.1f} GeV")

# ============================
# IC KERNEL (Blumenthal-Gould mono-field)
# ============================

me_GeV  = 0.000511     # electron rest mass
sigma_T = 6.652e-25    # [cm^2] (使っていないが、拡張用)
c_cgs   = 2.998e10     # [cm/s]

def ic_kernel_mono(Eg, Ee, eps_GeV):
    """
    単色 photon 場 (eps_GeV) に対する IC kernel（規格化なし）。
    Blumenthal & Gould の形を簡略化：
    dN/dEγ dt ∝ F(q, Γ) / (γ^2 ε)
    """
    if Eg <= 0 or Ee <= Eg:
        return 0.0
    gamma = Ee / me_GeV
    if gamma <= 1.0:
        return 0.0

    Gamma = 4.0 * gamma * eps_GeV / me_GeV
    if Gamma <= 0:
        return 0.0

    q = Eg / (Gamma * (Ee - Eg))
    if (q <= 0.0) or (q > 1.0):
        return 0.0

    term1 = 2.0 * q * np.log(q)
    term2 = (1.0 + 2.0*q) * (1.0 - q)
    term3 = 0.5 * (Gamma*q)**2 * (1.0 - q) / (1.0 + Gamma*q)
    F = term1 + term2 + term3

    return F / (gamma**2 * eps_GeV)

# photon fields: CMB + IR + Optical
photon_fields = [
    {"name": "CMB",     "eps_eV": 6e-4, "weight": 1.0},
    {"name": "IR",      "eps_eV": 1e-2, "weight": 1.2},
    {"name": "Optical", "eps_eV": 1.0,  "weight": 3.0},
]

# γ線エネルギー範囲（0.05–300 GeV）
Eg_min = 5e-2
Eg_max = 3e2
n_Eg   = 250
Eg_grid = np.logspace(np.log10(Eg_min), np.log10(Eg_max), n_Eg)

def IC_spectrum_kernel(Eg_grid, E_grid, nE, eps_eV):
    eps_GeV = eps_eV * 1e-9
    I = np.zeros_like(Eg_grid)
    for i, Eg in enumerate(Eg_grid):
        kernel_vals = []
        for Ee, ne_val in zip(E_grid, nE):
            k = ic_kernel_mono(Eg, Ee, eps_GeV)
            kernel_vals.append(ne_val * k)
        kernel_vals = np.array(kernel_vals)
        I[i] = np.trapz(kernel_vals, E_grid)
    return I

def IC_total_multifield(Eg_grid, E_grid, nE):
    comps = {}
    total = np.zeros_like(Eg_grid)
    for field in photon_fields:
        I_f = IC_spectrum_kernel(Eg_grid, E_grid, nE, field["eps_eV"])
        I_f *= field["weight"]
        comps[field["name"]] = I_f
        total += I_f

    if total.max() > 0:
        total_norm = total / total.max()
        for k in comps:
            comps[k] = comps[k] / total.max()
    else:
        total_norm = total
    return total_norm, comps

# ===== γ線スペクトル計算（TFGRなし / 各半径）=====
I_no_tot, _ = IC_total_multifield(Eg_grid, E_grid, nE_no)
I_tfgr_tot  = {}
I_tfgr_comp = {}
for r_kpc in radii_kpc:
    I_tot, comps = IC_total_multifield(Eg_grid, E_grid, nE_tfgr[r_kpc])
    I_tfgr_tot[r_kpc]  = I_tot
    I_tfgr_comp[r_kpc] = comps

# ============================
# PLOTS
# ============================

# 1) 電子比：各半径
plt.figure(figsize=(8,5))
for r_kpc in radii_kpc:
    ratio_e = nE_tfgr[r_kpc] / (nE_no + 1e-40)
    plt.plot(E_grid, ratio_e, label=f"{r_kpc:.0f} kpc")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Electron energy E [GeV]")
plt.ylabel("n_e(TFGR)/n_e(no TFGR)")
plt.title("Spectral hardening by TFGR (multiple radii)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("final_electron_ratio_multiR.png")
plt.close()

# 2) TFGRあり：各半径の total γ線スペクトル
plt.figure(figsize=(8,5))
plt.plot(Eg_grid, I_no_tot, label="no TFGR", color="gray", ls="--")
for r_kpc in radii_kpc:
    plt.plot(Eg_grid, I_tfgr_tot[r_kpc], label=f"{r_kpc:.0f} kpc (TFGR)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Gamma-ray energy Eγ [GeV]")
plt.ylabel("Iγ (normalized)")
plt.title("Total IC gamma-ray spectra (CMB+IR+Optical, kernel)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("final_gamma_total_multiR.png")
plt.close()

# 3) γ線比：各半径
plt.figure(figsize=(8,5))
for r_kpc in radii_kpc:
    ratio_g = I_tfgr_tot[r_kpc] / (I_no_tot + 1e-40)
    plt.plot(Eg_grid, ratio_g, label=f"{r_kpc:.0f} kpc")
plt.xscale("log")
plt.xlabel("Gamma-ray energy Eγ [GeV]")
plt.ylabel("Iγ(TFGR)/Iγ(no TFGR)")
plt.title("TFGR-induced excess in total IC gamma-rays (multiR)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.savefig("final_gamma_ratio_multiR.png")
plt.close()

print("Saved:",
      "final_electron_ratio_multiR.png,",
      "final_gamma_total_multiR.png,",
      "final_gamma_ratio_multiR.png")

# ============================================
# STEP 4: Fermi gll_iem_v07 (PRIMARY) vs TFGR
# ============================================
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

iem_path = "gll_iem_v07.fits"
hdul = fits.open(iem_path)
hdul.info()

# PRIMARY: shape = (28, 1441, 2880)
cube = hdul[0].data.astype(float)
print("PRIMARY cube shape:", cube.shape)

# --- Energy axis (HDU1) ---
E_table = hdul[1].data
E_ic = np.array(E_table["energy"], dtype=float) * 1e-3  # MeV→GeV

nE = cube.shape[0]
assert len(E_ic) == nE, f"Energy axis mismatch: E_ic={len(E_ic)}, cube_nE={nE}"

# --- lat/lon grids ---

hdr = hdul[0].header
nE, ny, nx = cube.shape

lon0 = hdr["CRVAL1"]
lat0 = hdr["CRVAL2"]
d_lon = hdr["CDELT1"]
d_lat = hdr["CDELT2"]
pix0_lon = hdr["CRPIX1"]
pix0_lat = hdr["CRPIX2"]

lons = lon0 + (np.arange(nx) + 1 - pix0_lon) * d_lon
lats = lat0 + (np.arange(ny) + 1 - pix0_lat) * d_lat
LON, LAT = np.meshgrid(lons, lats)

# --- halo region mask ---
b_cut = 20.0
mask = np.abs(LAT) > b_cut

# --- observed diffuse spectrum ---
I_obs = []
for iE in range(nE):
    plane = cube[iE, :, :]     # (lat, lon)
    values = plane[mask]
    I_obs.append(np.mean(values))

I_obs = np.array(I_obs)
I_obs_norm = I_obs / np.max(I_obs)

# --- TFGR model interpolation ---
R_comp = 50.0
I_tfgr = I_tfgr_tot[R_comp]          # from your code
I_tfgr_interp = np.interp(E_ic, Eg_grid, I_tfgr)
I_tfgr_norm = I_tfgr_interp / np.max(I_tfgr_interp)

# --- Figure 1: shape comparison ---
plt.figure(figsize=(8,5))
plt.plot(E_ic, I_obs_norm, "o-", label="Fermi diffuse (|b|>20°)")
plt.plot(E_ic, I_tfgr_norm, "s-", label=f"TFGR (R={R_comp:.0f} kpc)")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Eγ [GeV]")
plt.ylabel("Normalized intensity")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("step4_primary_vs_tfgr.png")
plt.close()

# --- Figure 2: ratio ---
ratio = I_obs_norm / (I_tfgr_norm + 1e-40)
plt.figure(figsize=(8,5))
plt.plot(E_ic, ratio, "o-")
plt.xscale("log")
plt.xlabel("Eγ [GeV]")
plt.ylabel("Fermi / TFGR")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("step4_primary_over_tfgr_ratio.png")
plt.close()

hdul.close()

print("Saved:")
print(" step4_primary_vs_tfgr.png")
print(" step4_primary_over_tfgr_ratio.png")

# ============================================
# STEP 5_excess: 低エネルギーでベースラインを決め、
#                残差（excess）に TFGR をフィット
# ============================================

E0_ref = 1.0  # GeV

# --- 5.1 低エネルギーで baseline パワーローをフィット ---
# 1–3 GeV を「TFGR ほぼ無視できる領域」として使用
E_base_min = 1.0
E_base_max = 3.0
mask_base = (E_ic >= E_base_min) & (E_ic <= E_base_max)

E_base = E_ic[mask_base]
Y_base = I_obs[mask_base]

# log-log 直線フィット: log Y = log A0 - gamma * log(E/E0)
x = np.log(E_base / E0_ref)
y = np.log(Y_base)

p = np.polyfit(x, y, 1)  # y ≈ p[0]*x + p[1]
gamma_base = -p[0]
A0_base = np.exp(p[1])

print("\n=== Baseline (low-E) fit ===")
print(f"  gamma_base = {gamma_base:.3f}")
print(f"  A0_base    = {A0_base:.3e}")

# ベースラインを全エネルギーで計算
I_base_full = A0_base * (E_ic / E0_ref) ** (-gamma_base)

# --- 5.2 残差スペクトル I_res = I_obs - I_base_full ---
I_res_full = I_obs - I_base_full

# 残差を見たいエネルギー範囲を設定（例: 5–300 GeV）
E_res_min = 5.0
E_res_max = 300.0
mask_res = (E_ic >= E_res_min) & (E_ic <= E_res_max)

E_res = E_ic[mask_res]
Y_res = I_res_full[mask_res]

# --- 5.3 残差に TFGR をフィット（振幅 A1 だけ） ---
R_list = [20.0, 50.0, 100.0]

best_R_ex = None
best_A1_ex = None
best_chi2_ex = np.inf

for R in R_list:
    I_tfgr_full = I_tfgr_tot[R]              # Eg_grid 上
    I_tfgr_res = np.interp(E_res, Eg_grid, I_tfgr_full)

    # 全ゼロ回避
    if np.all(I_tfgr_res == 0):
        continue

    # 最小二乗解 A1 = (Σ Y_res * T) / (Σ T^2)
    T = I_tfgr_res
    A1 = np.sum(Y_res * T) / np.sum(T ** 2)

    # 「excess を説明する」ので A1<0 は意味が無い → 0 にクリップ
    if A1 < 0:
        A1 = 0.0

    Y_model_res = A1 * T
    chi2 = np.sum((Y_res - Y_model_res) ** 2)

    print(f"[excess fit] R={R:4.1f} kpc, A1={A1:.3e}, chi2={chi2:.3e}")

    if chi2 < best_chi2_ex:
        best_chi2_ex = chi2
        best_R_ex = R
        best_A1_ex = A1

print("\n=== TFGR excess fit result ===")
print(f"  Best R_ex  = {best_R_ex:.1f} kpc")
print(f"  Best A1_ex = {best_A1_ex:.3e}")
print(f"  Min chi2_ex= {best_chi2_ex:.3e}")

# --- 5.4 ベストフィット TFGR を足した total モデルを描画 ---

if best_R_ex is not None:
    I_tfgr_best_full = I_tfgr_tot[best_R_ex]
    I_tfgr_best_ic = np.interp(E_ic, Eg_grid, I_tfgr_best_full)
    I_model_total = I_base_full + best_A1_ex * I_tfgr_best_ic

    # (a) Fermi vs baseline vs baseline+TFGR
    plt.figure(figsize=(8,5))
    plt.loglog(E_ic, I_obs, "ko", label="Fermi diffuse (|b|>20°)")
    plt.loglog(E_ic, I_base_full, "--", label=f"Baseline PL (γ={gamma_base:.2f})")
    if best_A1_ex > 0:
        plt.loglog(E_ic, best_A1_ex * I_tfgr_best_ic, ":", 
                   label=f"TFGR excess (R={best_R_ex:.0f} kpc)")
    plt.loglog(E_ic, I_model_total, "-", label="Baseline + TFGR (excess fit)")
    plt.xlabel("Eγ [GeV]")
    plt.ylabel("Intensity [model units]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("step5_excess_fit_components.png")
    plt.close()

    # (b) 残差 vs TFGR モデル
    plt.figure(figsize=(8,5))
    plt.semilogx(E_res, Y_res, "ko-", label="Residual: Fermi - baseline")
    if best_A1_ex > 0:
        plt.semilogx(E_res, best_A1_ex * np.interp(E_res, Eg_grid, I_tfgr_best_full),
                     "r--", label=f"TFGR fitted (R={best_R_ex:.0f} kpc)")
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Eγ [GeV]")
    plt.ylabel("Intensity residual")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("step5_excess_residuals.png")
    plt.close()

    print("Saved:")
    print("  step5_excess_fit_components.png")
    print("  step5_excess_residuals.png")
