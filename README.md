# Time-Field General Relativity (TFGR)
### A scale-dependent temporal field unifying atomic clocks, planetary systems, galaxies, and cosmology

This repository provides open, fully reproducible code and datasets for the  
**Time-Field General Relativity (TFGR)** framework — a scale-dependent temporal  
field model that quantitatively reproduces observations across more than  
**20 orders of magnitude**, including:

- Atomic clock deviations  
- GPS/LLR timing anomalies  
- Deep-space probe residuals  
- Galaxy rotation curves (SPARC)  
- Strong-lensing Einstein radii  
- Weak-lensing scale trends (KiDS/eFEDS)  
- Late-time cosmic acceleration  
- The jerk parameter j(0) ≈ 1.6  
- Convergence of all scales to a single critical length  
  **Lc ≈ 4 × 10⁹ m**

TFGR suggests that variations in measured time across the universe arise not  
from unknown mass-energy components (dark matter/dark energy), but from  
a **scale-dependent correction to the flow of proper time**.

---

## 🔷 Core Equation: Scale-Dependent Time Flow

TFGR models the proper-time increment as a function of observational scale **L**:

\[
\Delta t(L) = \Delta t_0 \left[ 1 + \left( \frac{L}{L_c} \right)^p \right]^q
\]

Same expression in plain ASCII (safe for any viewer):

```
Delta_t(L) = Delta_t0 * (1 + (L/Lc)**p)**q
```

Where:

- **Δt₀** : baseline proper-time unit  
- **L**   : observational scale  
- **Lc**  : universal critical length (~4 × 10⁹ m)  
- **p, q** : empirical scale-response exponents  

This single function yields consistent parameter estimates from:

- Quantum tunneling (sub-nm scale)  
- Optical lattice clocks (1–100 m)  
- Satellite timing (10⁶–10⁸ m)  
- Deep-space probes (10⁹–10¹³ m)  
- Galaxy rotation curves (10¹⁹–10²¹ m)  
- Strong/weak lensing (10²¹–10²⁴ m)  
- FRW cosmology and j(z) (10²⁶ m)

---

## 📁 Repository Structure

```
TFGR_Global_Analysis/
│
├── phase23A_sparc_rotation/        # SPARC galaxy rotation curves (TFGR vs data)
├── phase23D_strong_lensing/        # Strong gravitational lensing (Einstein radius)
├── phase23E_weak_lensing/          # Weak lensing scale trends (KiDS/eFEDS)
│
├── phase50_atomic_clocks/          # Optical clock TFGR fits (Sr/Yb/Mg)
├── phase51_gps_llr/                # GPS + LLR time-field tomography
├── phase52_spacecraft/             # Deep-space probe residuals (Voyager / NH)
│
├── phase102_energy_balance/        # Time-field energy-balance equation consistency
├── phase138_temporal_modes/        # Eigenmodes & stability of the temporal field
│
└── phase180_jerk_prediction/       # Jerk parameter j(0) ≈ 1.6 and future j(z)

```

---

## 🔧 Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Run a minimal example (galaxy rotation):

```
python phase23_galaxy_rotation/code/phase23_fit.py \
    --csv data/sample_sparc.csv
```

Run a cosmology example (j(z) reconstruction):

```
python phase102_cosmology/code/tfgr_cosmo_jz.py
```

All results (plots, CSV outputs) will be saved to each `results/` directory.

---

## 🔍 Scientific Motivation

TFGR proposes that discrepancies traditionally attributed to:

- dark matter (galaxy rotation, lensing),  
- dark energy (cosmic acceleration),  
- unexplained timing anomalies (atomic clocks, satellites, probes),

may instead arise from a **scale-dependent modification to proper time**.

Because:

- Photons carry **time stamps**, not intrinsic velocities.  
- Comparing distant systems requires **synchronization across scales**.  
- A small systematic drift in Δt(L) accumulates as apparent velocity or acceleration.  

Thus:

- Flat galaxy rotation curves  
- Strong-lensing Einstein radii  
- Hubble tension / jerk parameter  
- Timing anomalies  

can all emerge from the same underlying correction.

---

## 📜 Citation

A DOI will be assigned via Zenodo upon the first public release.  
Please cite this repository when using TFGR code, datasets, or equations.

---

## 🤝 Contributions & Discussion

Issues, discussions, and pull requests are welcome.  
Collaboration is especially encouraged from researchers in:

- astrophysics  
- general relativity  
- metrology  
- cosmology  
- dark-matter/modified-gravity theory  
- spacecraft navigation and timing
