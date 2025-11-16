import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --------------------------------------------
# Constants
# --------------------------------------------
G = 6.67430e-11
c = 2.99792458e8
kpc = 3.0857e19

# --------------------------------------------
# TFGR model parameters
# --------------------------------------------
V0 = 1e-8            # ↑ 高める（銀河スケールの有効項）
Phi_star = 7.5e-9
r_min_kpc = 0.05
r_max_kpc = 50.0

# --------------------------------------------
# Equations
# --------------------------------------------
def tfgr_equations(r, y):
    Phi, dPhi, Psi, dPsi = y
    if r < 1e-3 * kpc:
        r = 1e-3 * kpc
    dVdPhi = 2 * V0 * (Phi / Phi_star**2) * np.exp(-(Phi / Phi_star)**2)
    rho_phi = (0.5 * dPhi**2 + V0 * (1 - np.exp(-(Phi / Phi_star)**2))) / c**2
    ddPhi = dVdPhi - 2 * dPhi / r
    ddPsi = 4 * np.pi * G * rho_phi - 2 * dPsi / r
    return [dPhi, ddPhi, dPsi, ddPsi]

# --------------------------------------------
# Initial conditions
# --------------------------------------------
Phi0 = 0.0
dPhi0 = 1e-10     # ↑ わずかに押し上げる
Psi0 = 0.0
dPsi0 = 0.0
y0 = [Phi0, dPhi0, Psi0, dPsi0]

# --------------------------------------------
# Integration
# --------------------------------------------
r_vals = np.linspace(r_min_kpc * kpc, r_max_kpc * kpc, 400)
sol = solve_ivp(tfgr_equations, (r_vals[0], r_vals[-1]), y0,
                t_eval=r_vals, method="BDF", rtol=1e-6, atol=1e-10)

if not sol.success:
    print("Integration failed:", sol.message)
else:
    print(f"Integration OK. Steps = {len(sol.t)}")

Phi = sol.y[0]
Psi = sol.y[2]
r = sol.t

# --------------------------------------------
# Compute circular velocity
# --------------------------------------------
dPsi_dr = np.gradient(Psi, r)
v_circ = np.sqrt(np.maximum(r * dPsi_dr, 0.0))
v_circ_kms = v_circ / 1000

# --------------------------------------------
# Plot
# --------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(7, 10))

axs[0].plot(r / kpc, Phi / c**2, color="blue")
axs[0].set_xlabel("Radius [kpc]")
axs[0].set_ylabel(r"$\Phi_t / c^2$")
axs[0].set_title("TFGR Time-Field Profile")

axs[1].plot(r / kpc, Psi, color="green")
axs[1].set_xlabel("Radius [kpc]")
axs[1].set_ylabel(r"$\Psi_\Phi$")
axs[1].set_title("Effective Potential from Time Field")

axs[2].plot(r / kpc, v_circ_kms, color="red")
axs[2].set_xlabel("Radius [kpc]")
axs[2].set_ylabel("V [km/s]")
axs[2].set_title("TFGR Rotation Velocity (Self-consistent)")

plt.tight_layout()
plt.show()
