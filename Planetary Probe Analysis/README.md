# Time-Field General Relativity — Planetary Probe Analysis

**(Phase 30–35: Rosetta, New Horizons, Voyager 1/2, Arrokoth)**

This repository contains the full analysis pipeline, datasets, and generated results for testing **Time-Field General Relativity (TFGR)** at solar-system scales using multiple deep-space probes.
The included phases (30–35) evaluate whether the TFGR-predicted scale-dependent time-delay Δt(L) is visible in spacecraft residuals such as timing drift, velocity offsets, and distance-dependent anomalies.

The dataset spans **Rosetta, New Horizons, Voyager 1, Voyager 2**, and the unified scaling analysis that combines all spacecraft into a single Δt(L) curve.

## 📁 Directory Structure
```
Planetary Probe Analysis
├── phase30_Rosetta/
├── phase30B/
├── phase31_New Horizons/
├── phase32_Voyager1/
├── phase33_Voyager1/
├── phase33B_Voyager2/
├── phase34/
└── phase35/
```
## Each phase includes:

- Python source code
- CSV datasets (pre-processed probe telemetry)
- Output plots and fitted TFGR curves
- Phase-specific notes or test scripts

## 📦 Phase Descriptions
### Phase 30 – Rosetta / RPC-ICA

### Folder: phase30_Rosetta/

- Main scripts
  - phase_30_tfgr_fit.py
  - run_fit_free.py

- Dataset: rpcica_tfgr_ready_v2.csv
- Outputs:

  - phase30_out/rosetta_rpcica_v2_free_tfgr_fit.png

### Goal:
Fit TFGR’s Δt(L) model to the distance-dependent residuals obtained from Rosetta’s RPC-ICA instrument.
This phase tests whether **time-delay scaling** appears in near-comet spacecraft motion.

## Phase 30B – Rosetta (Supplementary Tests)

### Folder: phase30B/

Additional helper scripts and validation runs used during the Rosetta analysis.

## Phase 31 – New Horizons / Arrokoth

### Folder: phase31_New Horizons/

- Script: phase31_NewHorizons_tfgr_fit.py
- Dataset: nh_arrokoth_tfgr_ready.csv
- Outputs: phase31_out/…

### Goal:
Test TFGR scaling around the Pluto and Arrokoth flyby, checking whether TFGR reproduces the residual drift without invoking ad-hoc corrections.

## Phase 32 – Voyager 1 (Global Distance Sweep)

### Folder: phase32_Voyager1/

- Script: phase32_tfgr_fit.py
- Dataset: vg1_1977_2024_distance.csv
- Outputs: Voyager-1 TFGR fit plots

### Goal:
Use 40+ years of distance logs to evaluate whether TFGR time-delay behaves consistently in the **outer solar system** and near the heliopause.

## Phase 33 – Voyager 1 (Refined TFGR Fit)

### Folder: phase33_Voyager1/

A refined TFGR fit to Voyager-1 residuals with improved resolution and scanning.

## Phase 33B – Voyager 2 (Uranus Flyby)

### Folder: phase33B_Voyager2/

- Script: phase33B_tfgr_fit_uranus.py
- Dataset: vg2_uranus_19860124_distance.csv
- Outputs:

  - vg2_uranus_tfgr_fit_tfgr.csv
  - vg2_uranus_tfgr_fit_tfgr_vs_L.png
  - vg2_uranus_tfgr_fit_tfgr_vs_time.png
  - vg2_uranus_tfgr_fit_velocity.png

### Goal:
Analyze the 1986 Uranus flyby to test whether TFGR captures distance-dependent timing drift at planetary-flyby scales.

## Phase 34 – Unified Scaling Across All Spacecraft

### Folder: phase34/

- Script: phase34_tfgr_unified_scaling.py
- Outputs:

  - tfgr_unified_scaling.png
  - tfgr_unified_scaling_curve.csv

### Goal:
Combine Rosetta + New Horizons + Voyager1 + Voyager2 into a single Δt(L) curve.
This provides the key test that TFGR predicts a consistent critical scale:

**➤ Lc ≈ 4 × 10⁹ m (universal across all spacecraft)**

This phase is one of the strongest empirical demonstrations of TFGR scaling.

## Phase 35 – Time-Field Gradient Analysis

### Folder: phase35/

- Script: phase35_tfgr_field_gradient.py
- Outputs:

  - tfgr_field_gradient.csv
  - tfgr_field_gradient_phi.png
  - tfgr_field_gradient_gradient.png

### Goal:
Compute:

- The TFGR potential Φₜ(L)
- Its gradient dΦₜ/dL, representing temporal curvature
- The curvature transition scale

This provides a direct theoretical–observational link between TFGR parameters and spacecraft residuals.

## 🔧 How to Run the Code
### 1. Install Dependencies

Recommended Python version: 3.9–3.12

pip install numpy pandas matplotlib scipy

### 2. Example: Rosetta fit
cd phase30_Rosetta
python phase_30_tfgr_fit.py --csv rpcica_tfgr_ready_v2.csv

### 3. Unified scaling (Phase 34)
cd phase34
python phase34_tfgr_unified_scaling.py

### 4. Field-gradient computation (Phase 35)
cd phase35
python phase35_tfgr_field_gradient.py

## 📊 Key Scientific Findings
### ✔ All spacecraft datasets converge to the same TFGR scaling

The Δt(L) curves derived from Rosetta, New Horizons, Voyager 1, and Voyager 2 are
consistent with a single power-law scaling function.

### ✔ The TFGR critical scale is strongly supported

Across all independent missions:

**Lc ≈ 4 × 10⁹ m**

appears as the universal curvature/transition scale.

### ✔ TFGR fits outperform conventional models

AIC/BIC scores show significant improvements compared to:

- Constant residual models
- Linear distance models
- Exponential decay fits

### ✔ Outer-solar-system anomalies align with TFGR

Distance-dependent drifts in Voyager’s long-term data match TFGR predictions
without the need for extra parameters or unexplained forces.

## 📜 License

MIT License. All analysis scripts are free to use and modify.


