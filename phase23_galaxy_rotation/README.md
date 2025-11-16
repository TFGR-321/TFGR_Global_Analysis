# Phase23 – SPARC Galaxy Rotation Curves (TFGR Fit)

## 📌 Purpose
This phase tests whether the Time-Field General Relativity (TFGR) time-correction
\[
\Delta t(L) = \Delta t_0 \left[1 + (L/L_c)^p\right]^q
\]
can reproduce **galaxy rotation curves** without introducing dark matter.
We use high-quality SPARC rotation-curve data as the benchmark.

## 📁 Contents
This directory contains:

- **phase23A_sparc_fit_results.csv**  
  → TFGR best-fit parameters for each galaxy

- **phase23A_sparc_rotation_overlay.png**  
  → Observed vs TFGR rotation curve (representative sample)

- **phase23A_wp50_mass_model_comparison.png**  
  → Comparison between TFGR-based velocity and standard baryonic mass models

- **sparc_sample_input.csv**  
  → Cleaned SPARC input used for demonstration (no raw data required)

## 🧪 Method Summary
1. Import deprojected SPARC rotation-curve data  
2. Compute TFGR time-correction Δt(L) for each radial scale  
3. Convert TFGR-induced temporal corrections into effective circular velocity  
4. Fit parameters (Lc, p, q) globally across the sample  
5. Compare TFGR predictions vs observed rotation speeds

## ▶ Minimal Reproduction
Run the analysis:

```bash
python code/phase23A_fit.py \
    --csv data/sparc_sample_input.csv
