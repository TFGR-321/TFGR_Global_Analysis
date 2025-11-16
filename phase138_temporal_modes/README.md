# Phase138 – Temporal-Mode Stability Analysis of the TFGR Time Field

## 📌 Purpose
Phase138 investigates whether the Time-Field General Relativity (TFGR) time field
possesses **stable eigenmodes** and a well-defined **modal hierarchy**.  
This phase analyzes the *internal dynamics* of the time-field Φₜ and evaluates:

- stability of temporal modes  
- dominance of specific eigenmodes  
- convergence toward a steady-state distribution  
- existence of resonant transitions or bifurcations  

This is the most dynamical and mathematical part of the TFGR program, complementing
the observational tests from Phases 23–102.

---

## 📁 Contents

This directory includes:

- **phase138_loop_spectrum.csv**  
  Full loop-spectrum (modal power vs wavenumber)

- **phase138_mode_analysis_results.txt**  
  Summary of dominant mode, mode count, stability ratios

- **phase138_modal_power_distribution.png**  
  Modal power distribution P(k) vs k

- **phase138_loop_energy_flux.png**  
  Energy flux across modes

- **phase138_temporal_mode_overview.pdf**  
  Combined overview of modal hierarchy (optional)

All files are derived from the main analysis script.

---

## 🧪 Method Summary

### 1. Loop-Spectrum Input
The Phase138 analysis begins with a **loop spectrum** generated in earlier phases  
(typically Phase135–137).  
The loop spectrum captures the energy or amplitude distribution across modes:

\[
P(k) \quad \text{for} \quad k = 1, 2, \dots, N.
\]

This describes how much “temporal energy” is stored at each scale k.

### 2. Eigenmode Extraction
An eigenmode decomposition is performed using:

- modal power distribution,  
- cumulative spectral energy,  
- effective mode count \( N_\mathrm{eff} \),  
- resonance detection through curvature tests.

The following quantities are computed:

- **dominant mode** \( k_\mathrm{dom} \)  
- **fractional power** in the dominant mode  
- **effective number of modes**  
- **resonant transition points** where modal curvature changes sign

### 3. Stability Diagnostics
Stability is assessed using:

- modal curvature \( d^2P/dk^2 \)  
- cumulative power thresholds  
- flux balance across adjacent modes  
- sensitivity to parameter perturbations

A **positive-definite eigenvalue band** indicates stable temporal dynamics.

### 4. Key Findings
- The spectrum displays a clear **stable band** of modes  
- A single mode typically dominates with ~10–15% of total power  
- Modal hierarchy is smooth (no chaotic fragmentation)  
- Resonance points occur but do not cause instability  
- The TFGR time field behaves as a physically reasonable dynamical system

---

## ▶ Minimal Reproduction

Run the temporal-mode analysis:

```bash
python code/phase138_tfgr_temporal_mode_analysis.py \
    --spectrum_csv data/phase138_loop_spectrum.csv
