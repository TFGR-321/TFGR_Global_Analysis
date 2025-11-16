# Phase180 – Jerk Parameter Prediction and Future Cosmic Acceleration under TFGR

## 📌 Purpose
Phase180 investigates one of the most remarkable predictions of  
Time-Field General Relativity (TFGR):

\[
j(0) \approx 1.6,
\]

where \( j(z) \) is the **cosmic jerk parameter**, defined as:

\[
j(z) = \frac{\dddot{a}}{a H^3}.
\]

In the ΛCDM model:
- \( j(z) = 1 \) always (a fixed constant)

In TFGR:
- \( j(z) \) naturally deviates from 1  
- present-day \( j(0) \approx 1.6 \)  
- and increases toward \( j(z) \approx 2.1 \) in the far future

This provides a new way to probe late-time acceleration **without** invoking a cosmological constant or dark energy.

---

## 📁 Contents

This directory includes the key prediction outputs:

- **phase180_jz_reconstruction.csv**  
  → Reconstructed jerk parameter \( j(z) \) across redshift

- **phase180_jz_plot.png**  
  → Plot of \( j(z) \) vs \( z \), including \( j(0) \approx 1.6 \)

- **phase180_future_jz_prediction.png**  
  → TFGR prediction for the future evolution of \( j(z) \)

- **phase180_tfgr_jz_fit.txt**  
  → Numerical summary of the TFGR jerk reconstruction and best-fit parameters

These files arise from the main reconstruction script.

---

## 🧪 Method Summary

### 1. Time-field driven acceleration
In TFGR, the cosmic acceleration arises from the **temporal potential**:

\[
\Phi_t(L) = c^2 \frac{\Delta t(L)}{\Delta t_0},
\]

with the defining relation:

\[
\Delta t(L)
= \Delta t_0 \left[ 1 + \left( \frac{L}{L_c} \right)^p \right]^q.
\]

Changes in \( \Delta t(L) \) feed back into the expansion rate \( H(z) \),  
altering the acceleration and jerk.

### 2. Reconstruction of \( j(z) \)
The script reconstructs:

- \( H(z) \) from TFGR energy-balance trends (Phase102 connection)  
- \( q(z) = -\ddot a / (aH^2) \)  
- \( j(z) \) via numerical differentiation

### 3. Key findings
- Present day value:
  \[
  j(0) \approx 1.6 \quad (\text{TFGR prediction})
  \]
- ΛCDM fixed value:
  \[
  j_{\Lambda \rm CDM} = 1
  \]
- Future behavior:
  \[
  j(z \to -1) \rightarrow 2.1
  \]

This indicates **stronger late-time acceleration** than ΛCDM predicts,
but achieved through time-field dynamics rather than dark energy.

---

## ▶ Minimal Reproduction

Run the jerk reconstruction:

```bash
python code/phase180_tfgr_jerk_reconstruction.py \
    --csv data/phase180_jz_reconstruction.csv
