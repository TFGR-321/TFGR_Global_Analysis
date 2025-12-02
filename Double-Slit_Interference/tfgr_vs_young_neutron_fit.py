import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Physical parameters for thermal neutrons
# ==========================================================

# thermal neutron de Broglie wavelength (given)
lambda_n = 0.18e-9  # [m] 0.18 nm

print("=== Neutron parameters ===")
print(f"λ_thermal neutron = {lambda_n*1e9:.3f} nm")

# ==========================================================
# 2. Slit geometry (toy-realistic for neutron interferometry)
# ==========================================================

L0 = 5.0            # slit -> screen distance [m]
d  = 20.0e-6        # slit separation [m]  (20 µm)
a  = 5.0e-6         # slit width [m]       (5 µm)

# Standard Young fringe spacing:
delta_x = L0 * lambda_n / d   # [m]
print(f"Young fringe spacing Δx = {delta_x*1e3:.3f} mm")

k_std = 2*np.pi / delta_x     # spatial frequency for Young fringes

# ==========================================================
# 3. TFGR local derivative and matched ω_TF
# ==========================================================

# same TFGR local derivative as前回 (at L0 ~ 1 m を流用：toy-model)
d_dt_dL = 3.1341e-03   # [s/m]

# 要求される ω_TF（k_TF = k_std を満たすように）
# k_TF = ω_TF * (dΔt/dL)*(d/L0)
omega_TF = k_std * L0 / (d_dt_dL * d)
k_TF = omega_TF * d_dt_dL * d / L0

lambda_TF = 2*np.pi / k_TF

print("\n=== TFGR matched parameters (neutron) ===")
print(f"ω_TF     = {omega_TF:.3e} rad/s")
print(f"k_TF     = {k_TF:.3e} 1/m")
print(f"λ_TF     = {lambda_TF*1e3:.5f} mm (should match Δx)")
print(f"Difference (λ_TF - Δx) = {(lambda_TF - delta_x)*1e6:.5f} µm")

# ==========================================================
# 4. Build interference patterns
# ==========================================================

# transverse coordinate: ±5 fringe spacings
x = np.linspace(-5*delta_x, 5*delta_x, 2000)

# single-slit envelope (標準 Young と共通)
beta = (np.pi * a * x) / (lambda_n * L0)
I_envelope = (np.sinc(beta/np.pi))**2  # numpy.sinc: sin(pi x)/(pi x)

# Standard de Broglie / Young interference
I_std = I_envelope * (1 + np.cos(k_std * x)) / 2
I_std /= I_std.max()

# TFGR local-expansion interference
I_tf = I_envelope * (1 + np.cos(k_TF * x)) / 2
I_tf /= I_tf.max()

# x-axis in mm
x_mm = x * 1e3

# ==========================================================
# 5. Plot
# ==========================================================

plt.figure(figsize=(12,5))
plt.plot(x_mm, I_std, label=f"Standard neutron (λ={lambda_n*1e9:.2f} nm)", linewidth=2)
plt.plot(x_mm, I_tf, '--', label="TFGR local expansion (neutron)", linewidth=2)

plt.xlabel("x on screen (mm)")
plt.ylabel("normalized intensity")
plt.title("Neutron double-slit: de Broglie vs TFGR local-expansion interference")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
