import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ======================================================
# Phase 19 — TFGR Self-Consistent Fit vs Observed Data (Stable)
# ======================================================

# -------------------------------
# Physical constants
# -------------------------------
G   = 6.67430e-11          # m^3/kg/s^2
c   = 2.99792458e8         # m/s
kpc = 3.0857e19            # m

# -------------------------------
# Stable TFGR parameters
# -------------------------------
V0        = 120.0
Phi_star  = 1e-8
r_min_kpc = 1.0
r_max_kpc = 50.0

Phi0  = 0.0
dPhi0 = 1e-14
Psi0  = 0.0
dPsi0 = 0.0
y0    = [Phi0, dPhi0, Psi0, dPsi0]

# -------------------------------
# TFGR equations (stabilized)
# -------------------------------
def tfgr_equations(r, y):
    Phi, dPhi, Psi, dPsi = y

    if r < 1e-4 * kpc:
        r = 1e-4 * kpc

    # clamp potential range
    if abs(Phi) > 1e-5:
        Phi = np.sign(Phi) * 1e-5

    arg = (Phi / Phi_star)**2
    arg = np.clip(arg, -50, 50)

    dVdPhi = 2.0 * V0 * (Phi / Phi_star**2) * np.exp(-arg)
    rho_phi = (0.5 * dPhi**2 + V0 * (1 - np.exp(-arg))) / c**2

    # suppress runaway
    if not np.isfinite(dVdPhi):
        dVdPhi = 0.0
    if not np.isfinite(rho_phi):
        rho_phi = 0.0

    ddPhi = dVdPhi - 2.0 * dPhi / r
    ddPsi = 4.0 * np.pi * G * rho_phi - 2.0 * dPsi / r

    return [dPhi, ddPhi, dPsi, ddPsi]

# -------------------------------
# Integration setup
# -------------------------------
r_grid = np.linspace(r_min_kpc * kpc, r_max_kpc * kpc, 400)

sol = solve_ivp(
    tfgr_equations,
    (r_grid[0], r_grid[-1]),
    y0,
    t_eval=r_grid,
    method="Radau",       # ← BDF → Radau
    rtol=1e-6,
    atol=1e-8,
    max_step=0.2 * kpc,
)

if not sol.success:
    print("Integration failed:", sol.message)
    raise SystemExit

Phi = sol.y[0]
Psi = sol.y[2]
r   = sol.t

# 回転速度
dPsi_dr = np.gradient(Psi, r)
v_circ  = np.sqrt(np.maximum(r * dPsi_dr, 0.0))
v_circ_kms = v_circ / 1000.0

# -------------------------------
# Load observed data (NGC 3198 rotmod, tab-separated, comment-safe)
# -------------------------------
import pandas as pd
import numpy as np

cols = ["Rad","Vobs","errV","Vgas","Vdisk","Vbul","SBdisk","SBbul"]

df = pd.read_csv(
    "NGC3198.csv",
    sep=",",               # ← タブ区切り
    comment="#",            # ← 「#」で始まる行は全部スキップ！
    header=None,
    names=cols,
    encoding="utf-8-sig",   # ← BOM付きUTF-8対策
    engine="python"
).apply(pd.to_numeric, errors="coerce")

# データ整形
df = df.dropna(subset=["Rad","Vobs"])
df["errV"] = df["errV"].replace(0, np.nan).fillna(5.0)

r_obs = df["Rad"].values
Vobs  = df["Vobs"].values
Verr  = df["errV"].values
Vgas  = df["Vgas"].values
Vdisk = df["Vdisk"].values
Vbul  = df["Vbul"].values
Vbar  = np.sqrt(np.maximum(Vgas**2 + Vdisk**2 + Vbul**2, 0.0))

print("✅ 読み込み成功:", len(r_obs), "points")
print(df.head())

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.errorbar(r_obs, Vobs, yerr=Verr, fmt='o', color='black', label='Observed (NGC 3198)')
plt.plot(r_obs, Vbar, '--', color='gray', label='Baryonic')
plt.plot(r / kpc, v_circ_kms, 'r-', lw=2, label='TFGR Model')

plt.xlabel("Radius [kpc]")
plt.ylabel("Velocity [km/s]")
plt.title("TFGR Self-Consistent Fit vs Observed Data (Stable)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# Fit stats
# -------------------------------
v_interp = np.interp(r_obs, r/kpc, v_circ_kms)
mask = np.isfinite(Vobs) & np.isfinite(v_interp) & (Verr > 0)

if np.any(mask):
    rms  = np.sqrt(np.mean((Vobs[mask]-v_interp[mask])**2))
    chi2 = np.sum(((Vobs[mask]-v_interp[mask])/Verr[mask])**2) / np.sum(mask)
    print("\n--- TFGR Fit Statistics ---")
    print(f"V0       = {V0:.2e}")
    print(f"Phi_star = {Phi_star:.2e}")
    print(f"RMS error  : {rms:.2f} km/s")
    print(f"Reduced χ² : {chi2:.3f}")
    print("------------------------------------")
else:
    print("⚠ 有効な観測点がなく、RMS/χ²は計算されません。")
