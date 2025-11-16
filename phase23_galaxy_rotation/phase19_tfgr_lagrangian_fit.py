import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. Load data with manual header
# -----------------------------
colnames = ["Rad","Vobs","errV","Vgas","Vdisk","Vbul","SBdisk","SBbul"]
df = pd.read_csv("C:/Users/PC2FW08_U/Desktop/TFGR_GX/NGC3198.csv", sep=',', header=None, names=colnames)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna() 

r = df["Rad"].values
Vobs = df["Vobs"].values
Verr = df["errV"].values
Vgas = df["Vgas"].values
Vdisk = df["Vdisk"].values
Vbul = df["Vbul"].values

# -----------------------------
# 2. Compute baryonic component
# -----------------------------
Vbar = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)

# -----------------------------
# 3. Define TFGR-inspired model
# -----------------------------
def Vtfgr_model(r, V0, rc, n):
    return np.sqrt(Vbar**2 + V0**2 * (1.0 - np.exp(-(r/rc)**n)))

# -----------------------------
# 4. Fit model to observed data
# -----------------------------
p0 = [120.0, 13.0, 3.0]
bounds = ([50, 5, 0.5], [300, 30, 8])
popt, pcov = curve_fit(Vtfgr_model, r, Vobs, sigma=Verr, p0=p0, bounds=bounds, maxfev=20000)
V0_fit, rc_fit, n_fit = popt
perr = np.sqrt(np.diag(pcov))

print("\n--- TFGR Fit Results ---")
print(f"V0 = {V0_fit:.3f} ± {perr[0]:.3f} km/s")
print(f"rc = {rc_fit:.3f} ± {perr[1]:.3f} kpc")
print(f"n  = {n_fit:.3f} ± {perr[2]:.3f}")
print("-------------------------")

# -----------------------------
# 5. Derived potentials
# -----------------------------
Vmodel = Vtfgr_model(r, *popt)
Vphi = np.sqrt(np.maximum(Vmodel**2 - Vbar**2, 0))
Phi_t = 0.5 * (Vphi / 300000.0)**2   # dimensionless proxy Φ_t/c^2

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(8,6))
plt.errorbar(r, Vobs, yerr=Verr, fmt='o', color='black', label='Observed')
plt.plot(r, Vbar, '--', color='gray', label='Baryonic')
plt.plot(r, Vmodel, '-', color='red', lw=2, label='TFGR fit')
plt.xlabel("Radius [kpc]")
plt.ylabel("Velocity [km/s]")
plt.title("TFGR Lagrangian Fit: NGC 3198")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(r, Phi_t, 'b-', lw=2)
plt.xlabel("Radius [kpc]")
plt.ylabel("Φ_t / c²")
plt.title("Time-Field Potential Φ_t(r)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
