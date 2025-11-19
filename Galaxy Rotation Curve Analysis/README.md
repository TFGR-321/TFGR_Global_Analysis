# TFGR Galaxy Rotation Curve Analysis

### Time-Field General Relativity (TFGR) – Phase 22–23 Galaxy-Scale Validation

This repository contains all scripts, input data, and generated results used to evaluate  

**Time-Field General Relativity (TFGR)** on **galaxy rotation curves** across a wide sample  

of late-type and dwarf galaxies.  

The analysis corresponds to **Phase 22–23** of the TFGR validation program.

---

## Purpose of This Analysis

The goal of this phase is:

### To determine whether TFGR can reproduce galaxy rotation curves  

without invoking dark matter halos,  

and to compare the statistical evidence against ΛCDM halo models.

TFGR introduces a scale-dependent correction derived from  

the time-field potential:

[
Δt(L) = Δt_0 left\[ 1 + (L/L_c)^p right]^q.
]

This analysis tests whether the **effective radial acceleration\*\* induced by the  

time field can explain flat rotation curves.

---

## What the Scripts Do

### **1. `phase22B_tfgr_batch_fit.py`**

- Reads each galaxy CSV file  
- Computes TFGR-predicted rotation curve  
- Fits TFGR parameters per galaxy  
- Outputs:
- Best-fit parameters  
- AIC/BIC  
- Per-galaxy diagnostic plots (saved in `plots/`)

### **2. `phase23C_timefield_mass_scaling.py`**

- Computes the inferred **effective dynamical mass**  

from TFGR vs ΛCDM halo fits

- Generates global scaling relations



### **3. `phase23D_visualize_meff_scaling.py`**

- Creates visualization of Meff scaling, including:
- TFGR vs LCDM
- Residuals
- L/Lc dependence

---

## Key Results (Included in This Repository)

### **1. TFGR AIC/BIC performance**

File: `TFGR_AIC_BIC_comparison.png`

- Median **ΔAIC ≈ +216** in favor of TFGR  
- TFGR outperforms NFW/ISO ΛCDM halo models for the majority of galaxies  
- No dark matter halo is required in TFGR fits

### **2. Mass–scaling relation**

File: `TFGR_vs_LCDM_Meff_scaling.png`



- TFGR reproduces a natural ( M_{text{eff}} propto L^{alpha} ) scaling  

emerging from the time-field potential  

- LCDM requires separate free halo parameters per galaxy  
- TFGR explains data with a *single universal scale* ( L_c ≈ 4×10^9,mathrm{m} )

### **3. Per-galaxy fits**

Directory: `plots/`

Each plot includes:

- Observed rotation curve
  
- TFGR best-fit curve
  
- Residuals
  
- Parameter table
    
- χ², AIC, BIC

---

## Data Files

Each galaxy CSV contains:

- `r_kpc` – radial distance  
- `v_obs` – observed rotation velocity  
- `v_err` – measurement uncertainty  
- (Some files include luminosity or inclination metadata)

All preprocessing scripts assume this standard format.

---

## How to Run the Analysis

python phase22B\_tfgr\_batch\_fit.py
python phase23C\_timefield\_mass\_scaling.py
python phase23D\_visualize\_meff\_scaling.py

Dependencies:

-　Python ≥ 3.8
-　numpy
-　pandas
-　matplotlib
-　scipy

---

## Scientific Conclusion

The results in this repository support the following:

- TFGR accurately reproduces rotation curves without dark matter.
- Statistical evidence strongly favors TFGR over ΛCDM halo models.
- A single universal scale 

𝐿𝑐≈4×10^9𝑚 governs all galaxies.

This phase establishes TFGR as a viable alternative to dark matter

at galactic scales, forming a foundation for testing TFGR at cluster-scale

lensing and cosmic expansion.

---

## Citation

If you use this analysis, please cite:

Mitsui (2025).

"Time-Field General Relativity: Galaxy-Scale Validation (Phase 22–23)."
