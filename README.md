# Time-Field General Relativity (TFGR)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17622096.svg)](https://doi.org/10.5281/zenodo.17622096)

### A scale-dependent temporal field unifying atomic clocks, planetary systems, galaxies, and cosmology

This repository provides open, fully reproducible code and datasets for  
**Time-Field General Relativity (TFGR)** — a scale-dependent temporal-field  
framework that accurately reproduces observations across more than:

**20 orders of magnitude in scale**, from quantum tunneling to cosmic expansion.

TFGR successfully models:

- Atomic clock deviations  
- GPS/LLR timing anomalies  
- Deep-space probe residuals  
- Galaxy rotation curves (SPARC)  
- Strong gravitational lensing (Einstein radius)  
- Weak-lensing scale trends (KiDS/eFEDS)  
- Late-time cosmic acceleration  
- Jerk parameter prediction j(0) ≈ 1.6  
- Convergence of all scales to a single critical length  
  **Lc ≈ 4 × 10⁹ m**

TFGR suggests that discrepancies traditionally attributed to *dark matter* and  
*dark energy* may instead arise from a **scale-dependent correction to the flow  
of proper time**.

---

## 🔷 Core Equation: Scale-Dependent Time Flow

TFGR models the proper-time increment as a function of observational scale **L**:

$$
\Delta t(L) = \Delta t_0\,\left[1 + (L/L_c)^p\right]^q
$$

Where:

- **Δt₀** — baseline proper-time unit  
- **L** — observational scale  
- **Lc** — universal critical length (~4 × 10⁹ m)  
- **p, q** — empirical scale-response exponents  

This *single* scale-dependent correction yields consistent fits across:

- Quantum tunneling (10⁻¹⁰ m)  
- Optical lattice clocks (1–100 m)  
- GPS + LLR (10⁶–10⁸ m)  
- Deep-space probes (10⁹–10¹³ m)  
- Galaxy rotation curves (10¹⁹–10²¹ m)  
- Strong / weak lensing (10²¹–10²⁴ m)  
- Cosmic expansion & j(z) (10²⁶ m)

---

## 📁 Repository Structure

```
TFGR_Global_Analysis/
│
├── phase23_sparc_rotation/         # Galaxy rotation curves (SPARC, TFGR fits)
├── phase23D_strong_lensing/        # Strong lensing (Einstein radius analysis)
├── phase23E_weak_lensing/          # Weak-lensing scale trends (KiDS/eFEDS)
│
├── phase50_atomic_clocks/          # Optical clock TFGR fits (Sr/Yb/Mg)
├── phase51_gps_llr/                # GPS + LLR time-field tomography
├── phase52_spacecraft/             # Deep-space probes (Voyager / New Horizons)
│
├── phase102_energy_balance/        # Time-field energy-balance consistency
├── phase138_temporal_modes/        # Eigenmodes & stability analysis
│
└── phase180_jerk_prediction/       # Jerk parameter j(0) ≈ 1.6 and future j(z)
```

---

## 🔧 Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Run a minimal example (SPARC galaxy rotation):

```
python phase23_sparc_rotation/phase23_tfgr_sparc_fit.py \
    --csv sparc_sample_input.csv
```

Run cosmology energy-balance analysis:

```
python phase102_energy_balance/phase102_timefield_energy_balance.py
```

Run jerk-parameter reconstruction:

```
python phase180_jerk_prediction/phase180_jerk_reconstruction.py
```

All results (plots, CSV outputs) will appear inside each `results/` directory.

---

## 🔍 Scientific Motivation

TFGR provides a unified explanation for phenomena traditionally requiring  
*multiple separate mechanisms*:

- Galaxy rotation (dark matter)  
- Cosmic acceleration (dark energy)  
- Satellite / clock timing anomalies  
- Lensing mass discrepancies  

The key insight:

### **Photons carry time stamps — not velocities.**  
All astronomical and physical measurements ultimately rely on  
**comparisons of time across scales**.

Thus, if proper time slightly drifts depending on scale:

- galaxy rotation curves appear flat  
- Einstein radii shift  
- H(z) appears accelerated  
- j(0) rises to ≈ 1.6  
- deep-space probes show anomalous residuals  

All emerging from **the same Δt(L) correction**, with no new particles required.

---

## 📜 Citation

If you use TFGR code, datasets, or theoretical results, please cite:

**T. Mitsui (2025). TFGR Global Analysis. Zenodo.  
https://doi.org/10.5281/zenodo.17622096**

---

## 🤝 Contributions & Discussion

Contributions, issues, and discussions are welcome.

Researchers in the following fields are especially encouraged to collaborate:

- astrophysics  
- general relativity  
- metrology  
- cosmology  
- dark-matter/modified-gravity theory  
- spacecraft navigation & timing  

