# Phase102 – Cosmological Time-Field Energy Balance under TFGR

## 📌 Purpose
Phase102 evaluates the internal **energy consistency** of the Time-Field General
Relativity (TFGR) model in a cosmological context.  
Rather than fitting supernovae or measuring H(z), this phase tests whether the
TFGR time-field obeys a **self-consistent energy-balance relation** across cosmic
redshift.

The fundamental TFGR correction is:

\[
\Delta t(L) = \Delta t_0 \left[ 1 + \left( \frac{L}{L_c} \right)^p \right]^q,
\]

which induces a redshift-dependent temporal potential.  
Phase102 computes the resulting energy flux, field curvature, and effective
relaxation terms to verify whether:

\[
\text{Energy}_{\rm LHS}(z) \approx \text{Energy}_{\rm RHS}(z)
\]

holds over the observed redshift range.

---

## 📁 Contents

This directory contains the key processed results:

- **phase102_timefield_energy_balance.csv**  
  → Redshift, energy-balance LHS/RHS, residuals, and auxiliary terms

- **phase102_energy_balance_lhs_rhs.png**  
  → LHS vs RHS comparison across redshift

- **phase102_energy_balance_residual.png**  
  → Residual (LHS − RHS) showing consistency level

- **phase102_energy_balance_fit.txt**  
  → Summary of the best-fit TFGR parameters producing balance

These files are derived from the main analysis script below.

---

## 🧪 Method Summary

### 1. Time-field construction  
From the TFGR correction Δt(L), a corresponding **temporal potential** Φₜ(z) and  
its gradient **dΦₜ/dz** are derived.  
This yields a scale–redshift mapping where L(z) follows the cosmological distance.

### 2. Energy-balance equation  
The TFGR cosmological consistency condition is expressed as:

\[
\text{Energy}_{\rm LHS}(z)
= f_\mathrm{flux}(z)
+ f_\mathrm{curvature}(z),
\]

\[
\text{Energy}_{\rm RHS}(z)
= f_\mathrm{relaxation}(z),
\]

where LHS and RHS represent:

- **Energy flux of the time-field**
- **Field curvature term**
- **Relaxation / feedback term**

### 3. Numerical evaluation  
The script computes these quantities for each redshift point and evaluates:

- Agreement between LHS and RHS  
- Smoothness of energy flow across z  
- Parameter dependence (p, q, Lc)

### 4. Key result  
The energy-balance residual remains close to zero across the full z-range,
indicating that the TFGR field behaves like a **self-consistent cosmological
component**, without requiring Λ (cosmological constant) or exotic energy sources.

---

## ▶ Minimal Reproduction

Run the main energy-balance computation:

```bash
python code/phase102_timefield_energy_balance.py
