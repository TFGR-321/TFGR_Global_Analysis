#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFGR two-zone cosmic-ray transport model (GC vs Halo, complete version)

- Time-field: Δt(L) = Δt0 [1 + (L/Lc)^p]^q
- A_TFGR(L) is calibrated so that A(20 kpc) = -6.595e-11 1/s
  (same as in the previous halo analysis)
- 1D steady-state transport (leaky box) in each zone:
    d/dE{ [b_rad(E,L) - A_TFGR(L)*E] N_e } - N_e/t_esc(E,L) + Q(E) = 0
- Radiative losses: synchrotron + IC in Thomson approx
- Escape time: t_esc = t0(L) (E/1 GeV)^(-δ)
- IC gamma-ray spectra: simple δ-function approximation for each photon field

Outputs:
  - twozone_electron_spectra_full.png : E^2 N_e(E) for GC & Halo
  - twozone_gamma_spectra_full.png    : Eγ^2 dNγ/dEγ for GC & Halo
  - prints A_TFGR(L), κ_e, and some timescales
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Physical constants
# -------------------------------
c   = 2.99792458e8       # m/s
me  = 9.10938356e-31     # kg
eV  = 1.602176634e-19    # J
GeV = 1.0e9 * eV

# -------------------------------
# 2. TFGR time-field model
# -------------------------------
# Time-field parameters (same as you have been using)
dt0 = 1.0       # arbitrary (only ratios matter)
Lc  = 4.0e9     # m
p_tf = 0.2
q_tf = 1.3

def delta_t(L):
    """Δt(L) in arbitrary units (dt0 is just a scale)."""
    x = L / Lc
    return dt0 * (1.0 + x**p_tf)**q_tf

def d_delta_t_dL(L):
    """Numerical derivative dΔt/dL."""
    eps = 1e-3 * L
    if eps == 0:
        eps = 1e3  # avoid zero at L=0
    return (delta_t(L + eps) - delta_t(L - eps)) / (2*eps)

def calibrate_kappa_e():
    """
    Fix κ_e such that A_TFGR(20 kpc) = target_A = -6.595e-11 1/s,
    as in the previous halo analysis.
    """
    target_A = -6.595e-11  # 1/s at L = 20 kpc
    L_20kpc  = 20.0 * 3.086e19  # 1 kpc = 3.086e19 m
    ddt_dL   = d_delta_t_dL(L_20kpc)
    # A(L) = κ_e * c^2/Δt0 * dΔt/dL -> κ_e = A Δt0 / (c^2 dΔt/dL)
    kappa_e  = target_A * dt0 / (c**2 * ddt_dL)
    return kappa_e

kappa_e = calibrate_kappa_e()

def A_TFGR(L):
    """
    TFGR energy-gain coefficient A(L) [1/s],
    calibrated so that A(20 kpc) matches halo analysis.
    """
    return kappa_e * (c**2 / dt0) * d_delta_t_dL(L)

# -------------------------------
# 3. Zone definitions (GC vs Halo)
# -------------------------------
# We define two effective zones:
#   - GC   : R ~ 2 kpc
#   - Halo : R ~ 50 kpc
zones = {
    "GC": {
        "L_m":  2.0 * 3.086e19,   # 2 kpc in meters
        "B_uG": 50.0,             # magnetic field (μG)
        "U_rad_eVcm3": 10.0,      # photon energy density (eV/cm^3)
        "t0_esc_yr": 5.0e6,       # escape time at 1 GeV (yr)
    },
    "Halo": {
        "L_m": 50.0 * 3.086e19,   # 50 kpc in meters
        "B_uG": 2.0,              # μG
        "U_rad_eVcm3": 1.0,       # eV/cm^3 (CMB + faint IR/optical)
        "t0_esc_yr": 2.0e7,       # longer escape time (yr)
    }
}

delta_diff = 0.3  # diffusion index δ

# conversion: eV/cm^3 -> J/m^3  (1 eV = 1.602e-19 J, 1 cm^3 = 1e-6 m^3)
def eVcm3_to_Jm3(U_eVcm3):
    return U_eVcm3 * eV * 1.0e6

def UB_from_B_uG(B_uG):
    """Magnetic energy density U_B = B^2/(8π) in J/m^3, B in μG."""
    # 1 G = 1e-4 T -> 1 μG = 1e-10 T
    B_T = B_uG * 1.0e-10
    mu0 = 4.0e-7 * np.pi  # SI
    # U_B = B^2 / (2 μ0)
    return B_T**2 / (2.0 * mu0)

# Radiative loss coefficient b_rad(E) = b2 * E^2 (E in GeV, t in s)
# b2 ∝ (U_rad + U_B)
def b2_coeff(U_tot_Jm3):
    """
    Return b2 in units [1/s/GeV], using Thomson approximation.
    dE/dt = -(4/3) σ_T c (U_tot) (E/m_e c^2)^2
    but we work with E in GeV -> we absorb constants.
    """
    sigma_T = 6.6524587158e-29  # m^2
    mec2    = me * c**2         # J
    # dE/dt (J/s) = -(4/3) σ_T c U_tot (E_J/mec2)^2
    # convert to GeV/s: divide by GeV, E_J = E_GeV * GeV
    # => dE_GeV/dt = -C * U_tot * E_GeV^2
    C = (4.0/3.0) * sigma_T * c / (mec2**2) * (GeV**2 / GeV)
    return C * U_tot_Jm3

def t_esc(E_GeV, zone):
    """Energy-dependent escape time [s] for a given zone."""
    t0_yr = zone["t0_esc_yr"]
    t0_s  = t0_yr * 3.154e7
    return t0_s * (E_GeV)**(-delta_diff)

# -------------------------------
# 4. Source (injection) spectrum
# -------------------------------
p_inj  = 2.2
E_cut  = 5.0e3  # 5 TeV cutoff

def Q_inj(E_GeV):
    """Injection spectrum Q(E) ∝ E^{-p_inj} exp(-E/E_cut)."""
    return E_GeV**(-p_inj) * np.exp(-E_GeV / E_cut)

# -------------------------------
# 5. Solve steady-state transport
# -------------------------------
def solve_steady_state(zone, E_min=0.1, E_max=1e4, nE=300):
    """
    Solve for N_e(E) in a given zone using finite-difference:
      d/dE{ [b_eff(E) N] } - N/t_esc(E) + Q(E) = 0
    with boundary condition N(E_max)=0, integrate downwards.
    """
    # energy grid (GeV), log-spaced, from high to low
    E = np.logspace(np.log10(E_min), np.log10(E_max), nE)
    # we integrate from high to low, so reverse index
    E = E[::-1]

    L      = zone["L_m"]
    B_uG   = zone["B_uG"]
    U_rad  = eVcm3_to_Jm3(zone["U_rad_eVcm3"])
    U_B    = UB_from_B_uG(B_uG)
    U_tot  = U_rad + U_B
    b2     = b2_coeff(U_tot)        # [1/s/GeV]
    A_L    = A_TFGR(L)              # [1/s]

    # arrays
    N = np.zeros_like(E)
    Q = Q_inj(E)

    # precompute times and coefficients
    tesc = t_esc(E, zone)           # [s]
    b_rad = b2 * E**2               # [GeV/s]
    # effective coefficient: loss - TFGR gain
    # dE/dt = -b_rad*E^2 + A_L*E  → here we treat only in the transport term
    b_eff = b_rad - A_L * E

    # integrate from highest E downwards
    for i in range(len(E) - 2, -1, -1):
        dE = E[i] - E[i+1]  # note: negative number (E decreasing)
        # finite difference: (b_eff_{i+1} N_{i+1} - b_eff_i N_i)/dE - N_i/t_esc_i + Q_i = 0
        # → solve for N_i
        num = -b_eff[i+1] * N[i+1] / dE - Q[i]
        den = -b_eff[i] / dE - 1.0/tesc[i]
        if den != 0:
            N[i] = num / den
        else:
            N[i] = 0.0

        if N[i] < 0:
            # numerical safety: prevent negative population
            N[i] = 0.0

    # reverse back to ascending E
    return E[::-1], N[::-1], {"A_L": A_L, "b2": b2, "U_tot": U_tot}

# -------------------------------
# 6. IC gamma-ray spectra (δ-approx)
# -------------------------------
# We use simple δ-approx: for each photon field with mean energy eps,
# Eγ ~ (4/3) γ^2 eps  ⇒  E ≈ sqrt(3 Eγ / (4 eps)) * m_e c^2
# We sum contributions from CMB + IR + optical with different eps, U_rad weights.

def IC_spectrum_delta(EG, E, N, fields):
    """
    Compute IC spectrum using δ-function approximation.

    fields: list of dicts with keys:
      - 'eps_eV'   : mean photon energy [eV]
      - 'weight'   : relative weight (e.g. U_rad fraction)
    EG: gamma-ray energies [GeV]
    E : electron energies [GeV]
    N : electron spectrum N(E)
    Returns: I(EG) (arb units)
    """
    I = np.zeros_like(EG)

    for fld in fields:
        eps_J   = fld["eps_eV"] * eV
        weight  = fld["weight"]
        # for each EG, find corresponding electron energy via Thomson approx
        # Eγ ≈ (4/3) γ^2 eps ⇒ γ^2 ≈ (3 Eγ)/(4 eps)
        # E_e = γ m_e c^2
        gamma2 = (3.0 * (EG*GeV)) / (4.0 * eps_J)
        gamma2[gamma2 <= 0] = 1.0
        Ee_J  = np.sqrt(gamma2) * me * c**2
        Ee_GeV = Ee_J / GeV

        # interpolate N(Ee)
        Ne = np.interp(Ee_GeV, E, N, left=0.0, right=0.0)
        # simple Jacobian factor dE/dEγ ∝ 1/sqrt(Eγ)
        I += weight * Ne / np.sqrt(EG + 1e-30)

    return I

# -------------------------------
# 7. Run model for both zones
# -------------------------------
if __name__ == "__main__":
    print("=== TFGR two-zone complete model ===")
    print(f"kappa_e (from halo calibration) = {kappa_e:.3e} [1/s] / (dimensionless dΔt/dL)")

    results = {}
    for name, z in zones.items():
        print(f"\n--- Zone: {name} ---")
        E, N, aux = solve_steady_state(z)
        results[name] = {"E": E, "N": N, "aux": aux}

        A_L = aux["A_L"]
        U_tot = aux["U_tot"]
        b2 = aux["b2"]
        print(f"L = {z['L_m'] / 3.086e19:.2f} kpc")
        print(f"A_TFGR(L) = {A_L:.3e} 1/s")
        print(f"U_tot (rad+mag) = {U_tot/eV/1e6:.2f} eV/cm^3")
        # characteristic times at 100 GeV
        E100 = 100.0
        b100 = b2 * E100**2
        t_rad_100 = E100 / b100
        t_esc_100 = t_esc(E100, z)
        t_tfgr_100 = 1.0 / abs(A_L) if A_L != 0 else np.inf
        print(f"t_rad(100 GeV)  ≈ {t_rad_100/3.154e7:.2e} yr")
        print(f"t_esc(100 GeV)  ≈ {t_esc_100/3.154e7:.2e} yr")
        print(f"t_TFGR(100 GeV) ≈ {t_tfgr_100/3.154e7:.2e} yr")

    # ---------------------------
    # 8. Plot electron spectra
    # ---------------------------
    plt.figure(figsize=(8,6))
    for name, col in zip(["GC", "Halo"], ["tab:blue", "tab:orange"]):
        E = results[name]["E"]
        N = results[name]["N"]
        plt.plot(E, E**2 * N / np.max(E**2 * N), label=f"{name} electrons", lw=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Electron energy $E$ [GeV]")
    plt.ylabel(r"$E^2 N_e(E)$ (normalized)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("twozone_electron_spectra_full.png", dpi=200)

    # ---------------------------
    # 9. IC gamma-ray spectra
    # ---------------------------
    EG = np.logspace(-1, 4, 250)  # 0.1 GeV – 10 TeV

    # Photon fields for each zone
    # GC: strong optical + IR + CMB
    fields_GC = [
        {"eps_eV": 1.0,  "weight": 0.6},  # optical
        {"eps_eV": 0.03, "weight": 0.3},  # IR
        {"eps_eV": 6e-4, "weight": 0.1},  # CMB
    ]
    # Halo: mostly CMB + weak IR
    fields_Halo = [
        {"eps_eV": 6e-4, "weight": 0.8},  # CMB
        {"eps_eV": 0.03, "weight": 0.2},  # faint IR
    ]

    plt.figure(figsize=(8,6))
    for name, fields, col in zip(
        ["GC", "Halo"],
        [fields_GC, fields_Halo],
        ["tab:blue", "tab:orange"]
    ):
        E = results[name]["E"]
        N = results[name]["N"]
        I = IC_spectrum_delta(EG, E, N, fields)
        # normalize by maximum for visual comparison
        plt.plot(EG, EG**2 * I / np.max(EG**2 * I),
                 label=f"{name} IC gamma", lw=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Gamma-ray energy $E_\gamma$ [GeV]")
    plt.ylabel(r"$E_\gamma^2\, dN_\gamma/dE_\gamma$ (normalized)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("twozone_gamma_spectra_full.png", dpi=200)

    print("\nSaved:")
    print("  twozone_electron_spectra_full.png")
    print("  twozone_gamma_spectra_full.png")
