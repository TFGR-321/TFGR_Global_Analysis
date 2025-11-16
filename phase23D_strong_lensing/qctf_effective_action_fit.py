import numpy as np
import matplotlib.pyplot as plt

# === Parameters (from strong/weak lens fits) ===
p, q, Lc = 0.58, 0.40, 1e22
L0 = 1.0
sc = np.log(Lc / L0)

# === Define beta function ===
def beta_phi(phi, p, q):
    return q * p * (1.0 - np.exp(-phi / q))

# === Define potential ===
def U_phi(phi, p, q, Z=1.0):
    return -Z * q * p * (phi + q * np.exp(-phi / q))

# === Integration grid ===
s = np.linspace(np.log(1e3/L0), np.log(1e26/L0), 600)
phi = np.zeros_like(s)
phi[0] = q * np.log(1 + np.exp(p * (s[0] - sc)))

for i in range(len(s)-1):
    ds = s[i+1] - s[i]
    phi[i+1] = phi[i] + ds * beta_phi(phi[i], p, q)

# === Plot flow and potential ===
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(s, phi)
plt.xlabel("ln L")
plt.ylabel("φ = ln(Δt/Δt₀)")
plt.title("QCTF Flow")

plt.subplot(1,2,2)
phi_grid = np.linspace(0, 5*q, 200)
plt.plot(phi_grid, U_phi(phi_grid, p, q))
plt.xlabel("φ")
plt.ylabel("U(φ)")
plt.title("Effective Potential")

plt.tight_layout()
plt.show()
