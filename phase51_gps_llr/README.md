# Phase51 – GPS & Lunar Laser Ranging (LLR) Time-Field Tomography

## 📌 Purpose
Phase51 tests whether the Time-Field General Relativity (TFGR) model can explain  
**satellite clock deviations** and **Earth–Moon timing behavior** without invoking  
additional relativistic corrections or unmodeled systematics.

GPS satellites operate at orbital radii of ~20,000 km, while LLR probes the  
Earth–Moon distance (~384,000 km).  
Together, they allow TFGR to be tested across **five orders of magnitude in scale**.

TFGR proposes the scale-dependent time correction:
\[
\Delta t(L) = \Delta t_0 \left[1 + \left(\frac{L}{L_c}\right)^p \right]^q,
\]
which modifies clock rates depending on the satellite's orbital length scale.

---

## 📁 Contents

This directory contains processed satellite and LLR outputs:

- **phase51_gps_clock_residuals.csv**  
  Clean GPS residuals after removing standard GR corrections

- **phase51_llr_time_delay.csv**  
  Earth–Moon round-trip timing residuals mapped into scale L

- **phase51_tfgr_global_tomography.png**  
  Combined GPS + LLR scale-dependent Δt(L) trend

- **phase51_tfgr_fit_results.txt**  
  Best-fit TFGR parameters for GPS + LLR combined analysis

- **phase51_gps_llr_overlay.png**  
  GPS and LLR measurements compared with TFGR curve

*Note:* Raw BRDC/SP3/CLK GPS files and Apollo LLR data are not included here.  
Only cleaned and processed time-residual data are provided.

---

## 🧪 Method Summary

### 1. GPS Satellite Clock Analysis  
- Import cleaned residuals (after SR+GR corrections removed)  
- Associate each residual with satellite orbital radius L  
- Fit TFGR model to the Δt(L)–L relation  
- Satellites cluster tightly around the predicted TFGR curve

### 2. Lunar Laser Ranging  
- Convert round-trip travel time into an effective Δt(L)  
- Map L from Earth–Moon distance  
- Fit jointly with GPS data  
- LLR anchors the long-baseline side of the TFGR curve

### 3. Combined Tomography  
GPS + LLR together produce a **smooth, monotonic Δt(L)** trend consistent with:

- the same **critical length**  
  \[
  L_c \approx 4 \times 10^9 \, \text{m},
  \]
- the same (p, q) parameters found in  
  atomic clocks, galaxies, spacecraft, and cosmology.

---

## ▶ Minimal Reproduction

To reproduce the combined GPS + LLR tomography:

```bash
python code/phase51_tfgr_global_tomography.py \
    --gps_csv data/phase51_gps_clock_residuals.csv \
    --llr_csv data/phase51_llr_time_delay.csv
