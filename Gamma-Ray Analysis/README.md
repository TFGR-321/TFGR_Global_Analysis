# TFGR Gamma-Ray Analysis

This repository contains scripts and figures demonstrating how Time-Field General Relativity (TFGR)
can simultaneously explain:

- The Galactic Center Excess (GCE)
- The Galactic Halo hard gamma-ray component
- Electron spectrum distortions from TFGR-driven acceleration
- A unified two-zone cosmic-ray transport model (GC + Halo)

## Contents

### Scripts
- `digitize_gce_40x40.py`
- `tfgr_gce_chi2_fit.py`
- `tfgr_gce_bump_additive.py`
- `tfgr_halo_gamma_final.py`
- `tfgr_twozone_complete.py`

### Figures
See `figures/` for:
- GCE fits (TFGR vs base models)
- TFGR bump residuals
- Electron and gamma-ray spectra in GC/Halo zones

## Summary
TFGR introduces a scale-dependent time-field correction Δt(L) that modifies electron energy evolution.
Using a single parameter κₑ calibrated from halo data, the model reproduces:

- The 1–3 GeV bump seen in the GCE
- The hard high-energy tail seen in halo gamma-ray data
- Electron spectral differences between GC and Halo environments

This repository demonstrates that TFGR provides the first unified explanation of multiple
multi-wavelength Galactic phenomena without invoking dark matter.

## How to Run
Scripts require Python 3.9+ and the following packages:

```
numpy
scipy
matplotlib
astropy
```

Run each script directly:

```
python tfgr_gce_chi2_fit.py
python tfgr_twozone_complete.py
```

## License
MIT License
