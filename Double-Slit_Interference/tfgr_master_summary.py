# ============================================================
#  TFGR κ SUMMARY (numbers only, no plots)
# ============================================================

# Official kappa values (the same as used in your published figures)
kappa_photon   = 5.041e+08      # photon 633 nm
kappa_electron = 5.817e+13      # electron 50 keV
kappa_neutron  = 1.773e+12      # thermal neutron

# wavelengths
lambda_photon   = 633e-9
lambda_electron = 5.485e-12
lambda_neutron  = 0.18e-9

# compute kappa * lambda
A_ph = kappa_photon   * lambda_photon
A_el = kappa_electron * lambda_electron
A_nt = kappa_neutron  * lambda_neutron

A = (A_ph + A_el + A_nt) / 3.0   # universal constant

# print summary
print("\n================ TFGR κ SUMMARY (numbers only) ================\n")
print(f"Photon (633 nm):")
print(f"   κ  = {kappa_photon:.3e}")
print(f"   κλ = {A_ph:.3e}\n")

print(f"Electron (50 keV):")
print(f"   κ  = {kappa_electron:.3e}")
print(f"   κλ = {A_el:.3e}\n")

print(f"Neutron (thermal):")
print(f"   κ  = {kappa_neutron:.3e}")
print(f"   κλ = {A_nt:.3e}\n")

print("---------------------------------------------------------------")
print(f"TFGR universal constant  A = <κλ> = {A:.3e}")
print("===============================================================\n")
