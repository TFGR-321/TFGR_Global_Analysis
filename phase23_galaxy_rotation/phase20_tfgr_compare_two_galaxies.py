import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Phase 20 — TFGR Comparison: NGC 3198 vs NGC 2403
# ======================================================

# -------------------------------
# 1. Fitted parameters (from phase19 results)
# -------------------------------
params = {
    "NGC3198": {"V0": 119.47, "rc": 13.64, "n": 3.51},
    "NGC2403": {"V0": 108.33, "rc": 7.16,  "n": 2.12},
}

# -------------------------------
# 2. Define TFGR velocity model
# -------------------------------
def v_tfgr(r, V0, rc, n):
    """Analytic TFGR component"""
    x = np.clip(r / rc, 0.0, 1e3)
    return V0 * np.sqrt(1.0 - np.exp(-x**n))

# -------------------------------
# 3. Radius grid and normalized scale
# -------------------------------
r = np.linspace(0, 25, 500)
r_norm_3198 = r / params["NGC3198"]["rc"]
r_norm_2403 = r / params["NGC2403"]["rc"]

V3198 = v_tfgr(r, **params["NGC3198"])
V2403 = v_tfgr(r, **params["NGC2403"])

# -------------------------------
# 4. Plot comparison in absolute scale
# -------------------------------
plt.figure(figsize=(8,6))
plt.plot(r, V3198, 'r-', label='NGC 3198 (rc=13.6 kpc)')
plt.plot(r, V2403, 'b--', label='NGC 2403 (rc=7.2 kpc)')
plt.xlabel("Radius r [kpc]")
plt.ylabel("TFGR component Vₜ [km/s]")
plt.title("TFGR Velocity Term — Absolute Scale Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Plot comparison in normalized scale (r/rc)
# -------------------------------
plt.figure(figsize=(8,6))
plt.plot(r_norm_3198, V3198/params["NGC3198"]["V0"], 'r-', label='NGC 3198')
plt.plot(r_norm_2403, V2403/params["NGC2403"]["V0"], 'b--', label='NGC 2403')
plt.xlabel("Normalized radius r/rc")
plt.ylabel("Normalized TFGR velocity Vₜ/V₀")
plt.title("TFGR Profile Scaling: NGC 3198 vs NGC 2403")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Print summary
# -------------------------------
print("\n--- TFGR Parameter Comparison ---")
for g, p in params.items():
    print(f"{g}: V0={p['V0']:.2f} km/s, rc={p['rc']:.2f} kpc, n={p['n']:.2f}")
print("---------------------------------")
