# Phase23D – Strong Gravitational Lensing (Einstein Radius) under TFGR

## 📌 Purpose
This phase tests whether the Time-Field General Relativity (TFGR) model can  
reproduce **Einstein radii** of strong-lensing galaxies **without requiring dark matter halos**.

Observed Einstein radii (θ_E) depend on:
- lens mass distribution  
- distance ratios  
- **time delays and photon travel time**

TFGR modifies **photon travel time** through the scale-dependent correction:
\[
\Delta t(L) = \Delta t_0 \left[1 + (L/L_c)^p\right]^q.
\]

This alters the inferred θ_E without changing the baryonic mass.

---

## 📁 Contents

This directory contains the key outputs:

- **phase23D_einstein_radius_fit.csv**  
  → TFGR-predicted θ_E vs observed θ_E (KiDS/SLACS-like sample)

- **phase23D_einstein_radius_plot.png**  
  → Scatter plot: observed vs TFGR-predicted Einstein radius

- **phase23D_lens_profile_demo.png**  
  → Example lens profile under TFGR time-field correction

- **phase23D_tfgr_lens_summary.txt**  
  → Summary of the fitting procedure and best-fit parameters

*Note:* Original KiDS/eFEDS lensing catalogs are not included due to licensing.  
Only *derived* CSV/plots needed to reproduce TFGR results appear here.

---

## 🧪 Method Summary

1. For each strong-lensing system (lens redshift z_L, source redshift z_S):  
   - Compute angular-diameter distances  
   - Compute standard GR θ_E_GR using baryonic mass

2. Apply time-field correction:
   \[
   \theta_E^{\rm TFGR}(L) = \theta_E^{\rm GR} \times f_{\Delta t}(L)
   \]
   where L is the lens scale (kpc → m conversion included)

3. Fit TFGR parameters (p, q, Lc) across the sample  
4. Compare TFGR predictions with observed θ_E

Observational trend:
- GR + baryons alone underpredict θ_E  
- TFGR raises θ_E by modifying Δt(L), matching the data  
- No dark matter halo is required in this phase

---

## ▶ Minimal Reproduction

```bash
python code/phase23D_einstein_fit.py \
    --csv data/phase23D_einstein_radius_fit.csv
