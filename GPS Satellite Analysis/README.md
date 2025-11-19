# GPS Satellite Time-Field Analysis (Phase 31–32)

This repository contains the full workflow, scripts, and results for the
**Time-Field General Relativity (TFGR) satellite analysis**, performed using
multi-station GPS clock residuals.  
The goal is to determine whether GPS satellite timing errors contain a
consistent, station-independent structure predicted by TFGR.

---

## Overview

TFGR predicts that satellite clock residuals should follow a characteristic
linear pattern for each satellite *i*:

\[
\Delta t_i(L) = A_i + b_i \, L
\]

where:

- **\( A_i \)** : time-field amplitude for satellite *i*  
- **\( b_i \)** : gradient term  
- **\( L \)** : satellite–receiver line-of-sight distance (meters)

If TFGR is correct, the estimated parameters \( A_i \) should exhibit:

1. Strong consistency across different ground stations  
2. Significant reduction in residual RMS after applying TFGR correction  
3. Large improvements in information criteria (ΔAIC ≫ 0)  

This repository includes the scripts necessary to reproduce these results.


---

## How to Run

### **1. Run satellite-by-satellite TFGR fitting (Phase 31)**

```bash
python phase31_satellite_correlation_v2.py \
    --csv AJAC_phase30.csv \
    --out_prefix phase31_AJAC \
    --out_dir results \
    --plot

Repeat for ALIC, ANK2, and MIZU.

Outputs:

Cleaned TFGR-fit CSVs

Satellite-wise parameter tables (A_sat, b_sat, RMS_before, RMS_after, ΔAIC, R²)

Diagnostic plots for every satellite

---

### **2. Combine results across stations**

All station summaries are merged into a single file:

results/phase31_full_combined.csv

This file is used for multi-station comparison of A_sat.

---

### **3. Run station-level correlation with latitude (Phase 32)**

python phase32_latitude_correlation.py \
    --csv results/phase31_full_combined.csv \
    --out results/phase32_station_summary.csv

Outputs:

Mean / Std / Median of A_sat per station

Average R² and ΔAIC

Latitude correlation plot

figures/phase32_GPS_lat_vs_mean_A.png

---

## Key Findings

### 1. TFGR-fitting significantly improves satellite residuals

Across ~96 satellites from 3 stations:

RMS decreases by 1.0–3.5 m

ΔAIC improvements are consistently +20 to +80

R² typically 0.97–0.99

### 2. Time-field amplitude 𝐴𝑖 is consistent across stations

For AJAC, ALIC, ANK2:

Mean A_sat ≈ 1150–1280

MIZU (only 2 satellites available) shows near-zero A_sat due to insufficient data.

This cross-station agreement is strong evidence that the structure is
physical, not a station artifact.

### 3. Latitude dependence appears in station-level summary

A_sat tends to be slightly larger at lower latitudes.

Plot:
figures/phase32_GPS_lat_vs_mean_A.png

---

## Interpretation

Although individual satellite timing errors are extremely small
(picosecond-level), TFGR signatures emerge through:

Repeated linear patterns in residuals

Consistency across satellites

Agreement among geographically separated stations

This dataset provides statistical evidence that GPS satellites contain a
common time-field structure, as predicted by TFGR.

---

## Requirements

Python 3.9+

numpy, pandas, scipy

matplotlib

argparse

Install dependencies:

pip install -r requirements.txt

---

## Citation

If you use this analysis, please cite:

Mitsui, T. (2025).
Time-Field General Relativity: GPS Satellite Analysis (Phase 31–32).

---

## License

MIT License.
All analysis scripts are free to use and modify.