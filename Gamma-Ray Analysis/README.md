# TFGR Unified Gamma-Ray Analysis (Expert Version)

## 1. Scientific Motivation

Time-Field General Relativity (TFGR) proposes that the flow of time is scale-dependent,
parameterized by a universal correction function Δt(L).
This induces a spatial gradient dΔt/dL, acting as an effective acceleration term
in the evolution of particle energies.

The central hypothesis tested:

**A single TFGR-driven electron energy modulation parameter κₑ can simultaneously reproduce:**
- The Galactic Center Excess (GCE)
- Hard high-energy Galactic halo γ-rays
- Electron spectrum distortions
- GC/Halo differences in cosmic-ray transport

This yields a unified γ-ray phenomenology without dark matter.

---

## 2. TFGR Framework (Technical Summary)

Δt(L) = Δt₀ [1 + (L/Lc)^p]^q  
with Lc ≈ 4×10⁹ m (empirically consistent from atomic clocks → GPS → galaxies → cosmology).

Electron energy evolution:

dE/dt = -b(E)E² - E/τ_esc + κₑ (dΔt/dL) E

where:
- b(E): radiative losses (IC + synchrotron)
- τ_esc: escape timescale
- κₑ: TFGR–electron coupling
- dΔt/dL: time-field gradient

Predictions:
- Spectral hardening where |dΔt/dL| is large  
- Localized TFGR-induced bumps  
- GC/Halo differences from L, U_rad, B-field variations

---

## 3. Data Sets and Calibration

### 3.1 GCE Spectrum
Extracted using `digitize_gce_40x40.py`.
Fitted with:
- cutoff power-law (baseline)
- TFGR-driven bump spectrum

### 3.2 Galactic Halo
Pass 8 diffuse model (gll_iem_v07).
Halo γ-rays used to calibrate κₑ.

Calibrated value:

κₑ ≈ -2.1 × 10⁻⁹ s⁻¹ / (dimensionless dΔt/dL)

---

## 4. Scripts Included

- `digitize_gce_40x40.py`
- `tfgr_gce_chi2_fit.py`
- `tfgr_gce_bump_additive.py`
- `tfgr_halo_gamma_final.py`
- `tfgr_twozone_complete.py`

See repository documentation for detailed usage.

---

## 5. Principal Results

### 5.1 GCE Reproduction
TFGR produces a 1–3 GeV bump via competition between TFGR acceleration and radiative losses.

### 5.2 Halo Hardening
Using the same κₑ, the halo’s high-energy hard spectrum is reproduced.

### 5.3 Two-Zone Transport
`tfgr_twozone_complete.py` shows:
- GC zone: strong TFGR acceleration (t_TFGR ≪ t_rad, t_esc)
- Halo zone: moderate TFGR effects shaping high-energy IC components

This yields a consistent picture across Galactic scales.

---

## 6. Figures

Typical outputs in `figures/`:

- `gce_tfgr_chi2_fit_spectrum.png`
- `gce_tfgr_chi2_fit_residuals.png`
- `tfgr_gce_additive_fit.png`
- `tfgr_gce_additive_residuals.png`
- `twozone_gamma_spectra_full.png`
- `twozone_electron_spectra_full.png`

These illustrate TFGR's explanatory power and consistency.

---

## 7. Dependencies

```
python >= 3.9
numpy
scipy
matplotlib
astropy
```

---

## 8. Reproduction

```
python digitize_gce_40x40.py
python tfgr_gce_chi2_fit.py
python tfgr_gce_bump_additive.py
python tfgr_halo_gamma_final.py
python tfgr_twozone_complete.py
```

---

## 9. Interpretation

TFGR provides the first unified explanation for:
- GCE morphology
- Halo γ-ray hardening
- Electron spectrum bifurcation
- Scale-consistent κₑ

without requiring dark matter.

Future directions:
- Full-sky TFGR modeling
- AMS-02 e⁺/e⁻ connections
- Synchrotron constraints

---

## License
MIT License
