# Phase23E – Weak Gravitational Lensing (KiDS / eFEDS) under TFGR

## 📌 Purpose
Phase23E evaluates whether the Time-Field General Relativity (TFGR) framework can
reproduce **scale-dependent weak-lensing signatures** observed in surveys such as  
**KiDS, CFHTLenS, and eFEDS**, without invoking dark matter or modified clustering
parameters (σ₈, Ω_m).

Weak lensing provides a statistical measurement of cosmic shear and convergence κ(L),
making it a sensitive probe of scale-dependent physics.

---

## 📁 Contents

This directory contains derived datasets and figures:

- **phase23E_kids_shear_trend.csv**  
  TFGR-predicted shear amplitude versus scale, matched to KiDS-like trends

- **phase23E_kids_shear_plot.png**  
  Observed vs TFGR-predicted shear as a function of angular scale

- **phase23E_tfgr_lensing_fit.txt**  
  Summary of TFGR fit parameters (Lc, p, q)

- **phase23E_tfgr_shear_residual.png**  
  Scale-dependent residual between data and TFGR prediction

*Note:*  
Raw KiDS/eFEDS catalogs are not included for licensing reasons.  
Only processed, TFGR-compatible derived data are provided.

---

## 🧪 Method Summary

1. Weak-lensing observables such as  
   - shear γ(L),  
   - convergence κ(L),  
   - two-point statistics ξ₊ / ξ₋  

   exhibit **clear scale-dependent suppression** in real data (KiDS S₈ tension).

2. TFGR introduces a correction to photon propagation time:
   \[
   \Delta t(L) = \Delta t_0 \left[1 + \left(\frac{L}{L_c}\right)^p\right]^q,
   \]
   which modifies the effective lensing amplitude at large scales.

3. This correction naturally reduces lensing amplitude at tens of Mpc scales,
   bringing TFGR predictions closer to observed S₈-like values,
   **without altering matter density or clustering strength**.

4. The result:  
   TFGR can match the **low-amplitude cosmic shear** seen in KiDS/eFEDS
   without dark matter halos or modified ΛCDM parameters.

---

## ▶ Minimal Reproduction

Run the TFGR weak-lensing fit:

```bash
python code/phase23E_tfgr_weak_lensing_fit.py \
    --csv data/phase23E_kids_shear_trend.csv
