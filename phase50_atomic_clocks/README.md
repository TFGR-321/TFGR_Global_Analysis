# Phase50 – Atomic Clocks and TFGR Time-Flow Reconstruction

## 📌 Purpose
Phase50 evaluates whether the Time-Field General Relativity (TFGR) model can
reproduce **precision optical clock deviations** observed across different
baselines, heights, and frequency standards.  
These deviations are traditionally interpreted as gravitational redshift or systematics,
but TFGR introduces a unified scale-dependent correction:

\[
\Delta t(L) = \Delta t_0 \left[1 + \left(\frac{L}{L_c}\right)^p\right]^q.
\]

Atomic clocks allow testing TFGR at **meter-scale to kilometer-scale**, providing
a clean and controlled environment free from astrophysical uncertainties.

---

## 📁 Contents

This directory contains processed datasets and fit outputs:

- **phase50_clock_fit_results.csv**  
  TFGR best-fit parameters for each clock comparison

- **phase50_clock_overlay.png**  
  Observed vs TFGR-predicted frequency shift

- **phase50_global_scan_results.json**  
  Global parameter scan over (p, q, Lc)

- **phase50_tfgr_timeshift_plot.png**  
  Visualization of predicted Δt(L) at laboratory scales

- **phase50_clock_dataset_demo.csv**  
  Cleaned demo input for reproducing core results  
  (raw PTB/NIST/SYRTE data not included)

---

## 🧪 Method Summary

1. Optical lattice clocks provide frequency ratios  
   \( \nu_A / \nu_B \) measured at different:
   - elevations,  
   - baselines,  
   - transport configurations.  

2. Under TFGR, the proper time increment differs as a function of scale \( L \),
   modifying the observed frequency ratio:

   \[
   \frac{\nu_A}{\nu_B} \approx
   1 + \frac{\Delta t(L_A) - \Delta t(L_B)}{\Delta t_0}.
   \]

3. A nonlinear fit is performed to estimate global TFGR parameters  
   \( (p, q, L_c) \) across all datasets.

4. Results show a consistent convergence toward  
   **\( L_c \approx 4 \times 10^9 \, \mathrm{m} \)**,  
   the same critical length found in GPS, spacecraft, rotation curves,
   and cosmology.

---

## ▶ Minimal Reproduction

Run a basic atomic-clock TFGR fit:

```bash
python code/phase50_clock_fit.py \
    --csv data/phase50_clock_dataset_demo.csv
