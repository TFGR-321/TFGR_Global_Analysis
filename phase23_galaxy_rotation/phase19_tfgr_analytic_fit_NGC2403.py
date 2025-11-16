import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ======================================================
# Phase 19 — Analytic TFGR Fit for NGC 2403
# ======================================================

# -------------------------------
# 1. Load rotmod-style CSV
# -------------------------------
df = pd.read_csv(
    "NGC2403.csv",
    sep=",",
    header=0,
    encoding="utf-8-sig"
).apply(pd.to_numeric, errors="coerce")

df = df.dropna(subset=["Rad", "Vobs"])

r_obs  = df["Rad"].values
Vobs   = df["Vobs"].values
Verr   = df["errV"].values
Vgas   = df["Vgas"].values
Vdisk  = df["Vdisk"].values
Vbul   = df["Vbul"].values

# 誤差バーが 0 または NaN の点を補正
Verr = np.where((Verr <= 0) | ~np.isfinite(Verr), 5.0, Verr)

# バリオン速度
Vbar = np.sqrt(np.maximum(Vgas**2 + Vdisk**2 + Vbul**2, 0.0))

# バリオン補間関数
Vbar_interp = interp1d(r_obs, Vbar, kind="cubic",
                       fill_value="extrapolate", bounds_error=False)

# -------------------------------
# 2. Analytic TFGR model
# -------------------------------
def v_model_total(r, V0, rc, n):
    """
    r  : 半径 [kpc]
    V0 : TFGR 速度振幅 [km/s]
    rc : TFGR スケール半径 [kpc]
    n  : 立ち上がりの鋭さ
    """
    r = np.asarray(r)
    x = np.clip(r / rc, 0.0, 1e3)
    v_tfgr = V0 * np.sqrt(1.0 - np.exp(-x**n))
    v_bar  = Vbar_interp(r)
    return np.sqrt(np.maximum(v_bar**2 + v_tfgr**2, 0.0))

# -------------------------------
# 3. Curve fit TFGR parameters
# -------------------------------
fit_mask = (r_obs > 0.5)   # 中心 0.5 kpc を除外して安定化

r_fit    = r_obs[fit_mask]
V_fit    = Vobs[fit_mask]
Verr_fit = Verr[fit_mask]

p0     = [100.0, 7.0, 2.0]                 # 初期値
bounds = ([50.0,  2.0, 0.5], [300.0, 20.0, 6.0])

popt, pcov = curve_fit(
    v_model_total,
    r_fit, V_fit,
    sigma=Verr_fit,
    p0=p0,
    bounds=bounds,
    absolute_sigma=True,
    maxfev=20000
)

V0_best, rc_best, n_best = popt
perr = np.sqrt(np.diag(pcov))
V0_err, rc_err, n_err = perr

# -------------------------------
# 4. Evaluate model
# -------------------------------
r_grid = np.linspace(r_obs.min(), r_obs.max(), 500)
Vbar_grid   = Vbar_interp(r_grid)
Vtfgr_grid  = V0_best * np.sqrt(
    1.0 - np.exp(-np.clip(r_grid/rc_best, 0.0, 1e3)**n_best)
)
Vtot_grid   = np.sqrt(Vbar_grid**2 + Vtfgr_grid**2)

# -------------------------------
# 5. Fit statistics
# -------------------------------
Vmodel_at_obs = v_model_total(r_obs, V0_best, rc_best, n_best)
mask_stats = np.isfinite(Vobs) & np.isfinite(Vmodel_at_obs) & (Verr > 0)
rms  = np.sqrt(np.mean((Vobs[mask_stats] - Vmodel_at_obs[mask_stats])**2))
chi2 = np.sum(((Vobs[mask_stats] - Vmodel_at_obs[mask_stats]) /
               Verr[mask_stats])**2) / np.sum(mask_stats)

# -------------------------------
# 6. Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.errorbar(r_obs, Vobs, yerr=Verr, fmt="o", color="black",
             label="Observed (NGC 2403)", alpha=0.8)
plt.plot(r_grid, Vbar_grid, "--", color="gray", label="Baryonic")
plt.plot(r_grid, Vtfgr_grid, ":", color="green", label="TFGR term")
plt.plot(r_grid, Vtot_grid, "-", color="red", linewidth=2.0,
         label="Baryon + TFGR model")
plt.xlabel("Radius r [kpc]")
plt.ylabel("Rotation speed V [km/s]")
plt.title("NGC 2403 — Analytic TFGR Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Summary
# -------------------------------
print("\n--- Analytic TFGR Fit Results (NGC 2403) ---")
print(f"V0   = {V0_best:7.2f} ± {V0_err:6.2f} km/s")
print(f"rc   = {rc_best:7.2f} ± {rc_err:6.2f} kpc")
print(f"n    = {n_best:7.2f} ± {n_err:6.2f}")
print(f"RMS error   = {rms:6.2f} km/s")
print(f"Reduced χ²  = {chi2:6.3f}")
print("------------------------------------------------")
