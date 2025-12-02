# ============================================================
# File: tfgr_interference_3scale_comparison.py
# Purpose:
#   Compare TFGR local-expansion interference vs. standard
#   Young / de Broglie / neutron interference for 3 scales:
#   (Photon, Electron, Neutron)
#   + auto-generate κ comparison table.
# Author: ChatGPT TFGR-Lab
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ================================
# Common TFGR local derivative
# ================================
dDelta_t_dL = 3.1341e-3   # s/m (from TFGR)
pi = np.pi


# ================================
# Utility functions
# ================================
def standard_interference(x, k, d, L0):
    """Standard Young double-slit intensity"""
    beta = (pi * d * x) / (L0 * 1e3)   # convert x mm → meters
    return (np.sinc(beta / pi))**2 * (1 + np.cos(k * x*1e-3))

def tfgr_interference(x, k_TF, d, L0):
    beta = (pi * d * x) / (L0 * 1e3)
    return (np.sinc(beta / pi))**2 * (1 + np.cos(k_TF * x*1e-3))


# ================================
# Photon parameters
# ================================
lambda_ph = 633e-9   # m
d_ph = 0.25e-3        # m
L0_ph = 1.0            # m
Delta_x_ph = (lambda_ph * L0_ph) / d_ph   # fringe spacing
k_ph = 2*pi / Delta_x_ph

# TFGR matched
omega_ph = k_ph / (dDelta_t_dL * d_ph / L0_ph)
k_TF_ph = k_ph

x_ph = np.linspace(-12, 12, 2000)  # mm

I_std_ph = standard_interference(x_ph, k_ph, d_ph, L0_ph)
I_tf_ph = tfgr_interference(x_ph, k_TF_ph, d_ph, L0_ph)


# ================================
# Electron parameters
# ================================
lambda_e = 5.485e-12  # m
d_e = 1e-6            # m
L0_e = 1.0            # m
Delta_x_e = (lambda_e * L0_e) / d_e
k_e = 2*pi / Delta_x_e

omega_e = k_e / (dDelta_t_dL * d_e / L0_e)
k_TF_e = k_e

x_e = np.linspace(-0.03, 0.03, 2000)*1e3  # convert to mm

I_std_e = standard_interference(x_e, k_e, d_e, L0_e)
I_tf_e = tfgr_interference(x_e, k_TF_e, d_e, L0_e)


# ================================
# Neutron parameters
# ================================
lambda_n = 0.18e-9   # m
d_n = 20e-6          # m
L0_n = 5.0           # m
Delta_x_n = (lambda_n * L0_n) / d_n
k_n = 2*pi / Delta_x_n

omega_n = k_n / (dDelta_t_dL * d_n / L0_n)
k_TF_n = k_n

x_n = np.linspace(-0.25, 0.25, 2000)*1e3  # mm

I_std_n = standard_interference(x_n, k_n, d_n, L0_n)
I_tf_n = tfgr_interference(x_n, k_TF_n, d_n, L0_n)


# ================================
# Figure with 3 panels + κ table
# ================================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])

# --- Photon panel ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_ph, I_std_ph, label="Standard Young", lw=1.5)
ax1.plot(x_ph, I_tf_ph, '--', label="TFGR local", lw=1.5)
ax1.set_title("Photon (633 nm)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("normalized intensity")
ax1.legend()

# --- Electron panel ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x_e, I_std_e, label="Standard de Broglie", lw=1.5)
ax2.plot(x_e, I_tf_e, '--', label="TFGR local", lw=1.5)
ax2.set_title("Electron (50 keV)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("normalized intensity")
ax2.legend()

# --- Neutron panel ---
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(x_n, I_std_n, label="Standard neutron", lw=1.5)
ax3.plot(x_n, I_tf_n, '--', label="TFGR local", lw=1.5)
ax3.set_title("Neutron (thermal)")
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("normalized intensity")
ax3.legend()

# --- κ table ---
ax_table = fig.add_subplot(gs[:, 1])
ax_table.axis('off')

table_text = (
    " TFGR coupling κ summary\n"
    "-------------------------\n"
    f"Photon (633 nm):     κ ≈ {omega_ph/(2*np.pi):.3e}\n"
    f"Electron (50 keV):    κ ≈ {omega_e/(2*np.pi):.3e}\n"
    f"Neutron (thermal):    κ ≈ {omega_n/(2*np.pi):.3e}\n"
)

ax_table.text(0.05, 0.95, table_text, fontsize=14, va='top', family='monospace')

plt.tight_layout()
plt.show()
