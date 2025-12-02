import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Physical parameters for 50 keV electrons
# ==========================================================

m_e = 9.10938356e-31     # electron mass [kg]
h   = 6.62607015e-34     # Planck [J*s]
hbar = h/(2*np.pi)
eV  = 1.602176634e-19

E_kev = 50.0e3 * eV                     # 50 keV kinetic energy
p = np.sqrt(2 * m_e * E_kev)            # non-relativistic momentum
lambda_e = h / p                         # electron de Broglie wavelength

print("=== Electron parameters ===")
print(f"Electron KE     = 50 keV")
print(f"λ_deBroglie     = {lambda_e*1e12:.3f} pm")

# ==========================================================
# 2. Slit geometry (toy-realistic)
# ==========================================================

L0 = 1.0           # slit → screen [m]
d  = 1.0e-6        # slit separation [m]
a  = 0.1e-6        # slit width [m]

# Standard Young fringe spacing:
delta_x = L0 * lambda_e / d
print(f"Young fringe spacing Δx = {delta_x*1e3:.3f} mm")

k_std = 2*np.pi / delta_x

# ==========================================================
# 3. TFGR local derivative (example value from your results)
# ==========================================================

d_dt_dL = 3.1341e-03   # [s/m] from TFGR Δt(L) derivative at L0

# Required ω_TF to match k_std:
# k_TF = ω_TF * (dΔt/dL)*(d/L0)
omega_TF = k_std * L0 / (d_dt_dL * d)
k_TF = omega_TF * d_dt_dL * d / L0

print("\n=== TFGR matched parameters ===")
print(f"ω_TF     = {omega_TF:.3e} rad/s")
print(f"k_TF     = {k_TF:.3e}")
print(f"λ_TF     = {2*np.pi/k_TF*1e3:.5f} mm (should match Δx)")
print(f"Difference = {(2*np.pi/k_TF - delta_x)*1e6:.5f} µm")

# ==========================================================
# 4. Build patterns
# ==========================================================

# transverse coordinate (± 5 fringe widths)
x = np.linspace(-5*delta_x, 5*delta_x, 2000)

# Standard de-Broglie interference intensity
beta = (np.pi * a * x) / (lambda_e * L0)
I_envelope = (np.sinc(beta/np.pi))**2   # note: numpy sinc is sin(pi x)/(pi x)

I_std = I_envelope * (1 + np.cos(k_std * x)) / 2
I_std /= I_std.max()

# TFGR local expansion version
I_tf = I_envelope * (1 + np.cos(k_TF * x)) / 2
I_tf /= I_tf.max()

# convert x-axis to mm
x_mm = x * 1e3

# ==========================================================
# 5. Plot
# ==========================================================

plt.figure(figsize=(12,5))
plt.plot(x_mm, I_std, label=f"Standard de Broglie (λ={lambda_e*1e12:.2f} pm)", linewidth=2)
plt.plot(x_mm, I_tf, '--', label="TFGR local expansion", linewidth=2)

plt.xlabel("x on screen (mm)")
plt.ylabel("normalized intensity")
plt.title("Electron double-slit: de Broglie vs TFGR local-expansion interference")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
