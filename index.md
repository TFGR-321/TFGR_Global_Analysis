<!-- MathJax -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>athjax: true

# Time-Field General Relativity (TFGR)
### A scale-dependent temporal field unifying atomic clocks, planetary systems, galaxies, and cosmology  
**DOI:** [10.5281/zenodo.17622096](https://doi.org/10.5281/zenodo.17622096)

---

## Introduction

**Time-Field General Relativity (TFGR)** is a scale-dependent modification of general relativity in which  
the *flow of proper time* acquires a systematic dependence on the observational scale **L**.

Across more than **20 orders of magnitude**—from sub-nanometer quantum tunneling to cosmic expansion—  
TFGR reproduces a wide range of anomalies traditionally attributed to dark matter, dark energy, or  
instrumental timing errors.

These include:

- Optical lattice clock deviations  
- GPS/LLR timing residuals  
- Deep-space probe anomalies (Voyager, New Horizons)  
- Galaxy rotation curves (SPARC)  
- Strong gravitational lensing (Einstein radius)  
- Weak-lensing scale trends (KiDS/eFEDS)  
- Late-time cosmic acceleration  
- Jerk parameter prediction j(0) ≈ 1.6  

A striking feature emerging from all analyses is the convergence to a **universal critical length**

> **Lc ≈ 4 × 10⁹ m**

which governs the onset of scale-dependent temporal curvature.

---

## Core Equation: Scale-dependent proper time

TFGR models the scale-dependent proper-time increment as follows:

$$
\Delta t(L) = \Delta t_0 \left[ 1 + \left( \frac{L}{L_c} \right)^p \right]^q
$$

Where:

- **Δt₀** – baseline proper-time unit  
- **L** – observational scale  
- **Lc** – universal critical scale (~4×10⁹ m)  
- **p, q** – empirical scale-response exponents determined by data  

This single function explains the common structure seen in:

- flat galaxy rotation curves  
- lensing mass discrepancies  
- timing offsets in satellites and spacecraft  
- late-time cosmic acceleration  
- predicted present-day jerk parameter **j(0) ≈ 1.6**  

The time-field potential is:

$$
\Phi_t(L) = c^2 \frac{\Delta t(L)}{\Delta t_0},
$$

and its derivative \( d\Phi_t/dL \) produces the **temporal curvature** felt universally.

---

## The Universal Scale \( L_c \approx 4\times10^9\ \mathrm{m} \)

Independent datasets—spanning quantum to cosmological domains—reveal the same characteristic scale.

This **scale universality** is a key signature of the underlying physical time-field.

---

## Phase Overview

### **Phase 23A — SPARC Galaxy Rotation**
- TFGR fits outperform ΛCDM  
- ΔAIC ≈ **216** in favor of TFGR  
- Flat rotation curves reproduced without dark matter  

### **Phase 23D — Strong Lensing**
- Einstein radius offsets match TFGR curvature  

### **Phase 23E — Weak Lensing**
- KiDS/eFEDS shear trends consistent with TFGR scaling  

### **Phase 50 — Optical Lattice Clocks**
- Reveals same Lc ≈ 4×10⁹ m  

### **Phase 51 — GPS + LLR**
- Earth–satellite–Moon system probes L = 10⁶–10⁹ m  

### **Phase 52 — Deep-space Probes**
- Voyager / NH residuals match TFGR predictions  

### **Phase 102 — Cosmological Energy Balance**
- TFGR energy flux reproduces late-time acceleration  

### **Phase 138 — Temporal Eigenmodes**
- Confirms stability of the time-field  

### **Phase 180 — Jerk Parameter j(z)**
- Present-day prediction **j(0) ≈ 1.6**  
- Future limit **j(z→∞) ≈ 2.1**  

---

## Reproducibility

### Install dependencies

pip install -r requirements.txt

### Run SPARC galaxy rotation example

python phase23_sparc_rotation/phase23A_tfgr_sparc_fit.py
--csv sparc_sample_input.csv

### Run cosmology energy-balance example

python phase102_energy_balance/phase102_timefield_energy_balance.py

### Run jerk-parameter reconstruction

python phase180_jerk_prediction/phase180_jerk_reconstruction.py

Outputs are saved to each phase's `results/` directory.

---

## Citation

Please cite via:

**Zenodo DOI:**  
https://doi.org/10.5281/zenodo.17622096

---

## Links

- **GitHub Repository**  
  https://github.com/TFGR-321/TFGR_Global_Analysis

- **GitHub Discussions**  
  https://github.com/TFGR-321/TFGR_Global_Analysis/discussions

- **Zenodo Archive**  
  https://zenodo.org/doi/10.5281/zenodo.17622096

---

© 2025 TFGR Collaboration
