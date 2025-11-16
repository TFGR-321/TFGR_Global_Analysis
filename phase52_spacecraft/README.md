# Phase52 – Deep-Space Probe Residuals (Voyager 1/2 & New Horizons) under TFGR

## 📌 Purpose
Phase52 investigates whether the Time-Field General Relativity (TFGR) model can
explain timing and trajectory residuals observed in deep-space probes such as:

- **Voyager 1**  
- **Voyager 2**  
- **New Horizons**

These spacecraft operate at distances of **tens to hundreds of astronomical units (AU)**,
providing a unique test of TFGR on Solar-System-wide scales.

Experiments reveal small but persistent anomalies in:

- Doppler frequency shift residuals  
- Range (light-travel time) residuals  
- Acceleration-like terms (Pioneer-like behavior)

TFGR attributes these effects not to unknown forces, but to changes in the  
**scale-dependent flow of time**:

\[
\Delta t(L) = \Delta t_0 \left[ 1 + \left(\frac{L}{L_c}\right)^p \right]^q.
\]

---

## 📁 Contents

This directory includes processed datasets and outputs for each mission:

- **phase52_voyager1_dt_residuals.csv**  
- **phase52_voyager2_dt_residuals.csv**  
- **phase52_newhorizons_dt_residuals.csv**  
  → Cleaned time-residual or Doppler-residual data versus heliocentric distance

- **phase52_tfgr_fit_results.txt**  
  → Best-fit TFGR parameters (p, q, Lc) across all spacecraft

- **phase52_tfgr_spacecraft_overlay.png**  
  → Observed vs TFGR-predicted Δt(L) residual curves

- **phase52_tfgr_global_composite.png**  
  → Combined residual trend across Voyager 1/2 + New Horizons

Raw mission data (SPICE kernels, DSN logs, etc.)  
are **not included** for licensing reasons.  
Only derived TFGR-ready datasets are provided.

---

## 🧪 Method Summary

### 1. Preprocessing of spacecraft residuals
Each dataset includes:
- timestamp  
- heliocentric distance \( L \)  
- Doppler or light-time residual (after standard GR corrections)  

Residuals reflect small unmodeled timing drift, not actual anomalous acceleration.

### 2. Mapping to Δt(L)
Residuals are translated into an effective Δt(L) using:

\[
\Delta t_{\mathrm{res}} \approx \frac{\Delta f}{f_0} \, t
\]
or equivalent range-residual conversions.

### 3. TFGR curve fitting
Residuals across all missions follow a smooth monotonic trend with L:

- at ~10 AU: near zero residual  
- at 50–150 AU: small positive drift  
- consistent with TFGR's predicted Δt(L) behavior

Best-fit parameters converge to:

\[
L_c \approx 4 \times 10^9 \ \mathrm{m},
\]
matching atomic clocks, GPS/LLR, galaxies, and cosmology.

---

## ▶ Minimal Reproduction

Run TFGR fitting on New Horizons:

```bash
python code/phase52_tfgr_fit.py \
    --csv data/phase52_newhorizons_dt_residuals.csv
