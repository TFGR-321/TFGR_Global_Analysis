TFGR: Jerk-Based Cosmological Diagnostics (Phase 186–188)
(Time-Field General Relativity — SN/BAO constraints & future-universe jerk prediction)

🌌 Overview

This repository contains the analysis scripts and results for Phase 186–188 of the TFGR research program.
These phases focus on a critical cosmological diagnostic — the jerk parameter

$$ 
j(z) \equiv \frac{\dddot{a}}{a H^{3}}
 $$	​


which equals 1 for all redshifts in ΛCDM.
Any observational deviation from 𝑗(𝑧)=1 can therefore serve as a powerful discriminator between TFGR and ΛCDM.

Across these phases, we show that:

TFGR naturally predicts 𝑗(0)≈1.6(today)

And 𝑗(𝑧)→2.1 in the future universe (𝑧 → −1)

These predictions remain consistent with SN+BAO data

The best-fit TFGR model slightly outperforms ΛCDM in χ² and AIC

📁 Files included in this folder
phase186_tfgr_nTF_SN_BAO_scan.csv
phase186_tfgr_nTF_SN_BAO_best_summary.csv
phase187_nTF-0p6_future_Hqj_profile.csv
phase187_nTF-0p6_future_j_z.png
phase187_nTF-0p6_future_q_z.png
phase187_nTF-0p6_future_H_z.png
phase188_tfgr_j0_grid_tfgr_scan.csv
phase188_tfgr_j0_grid_lcdm_scan.csv
phase188_tfgr_j0_grid_chi2_vs_Om_m.png
...

🧭 Phase 186 — Joint SN+BAO Fit (n_TF Scan)
🔍 Purpose

Estimate the TFGR power-law index

$$ 
\rho_{\mathrm{TF}}(z) = \Omega_{\mathrm{TF},0} (1+z)^{n_{\mathrm{TF}}} 
$$

using combined SN (Pantheon) + BAO + H(z) data.

🏁 Results
| Parameter        | Best-fit     |
| ---------------- | ------------ |
| ( \Omega_{m0} )  | **0.34**     |
| ( n_{TF} )       | **−0.60**    |
| ( \Omega_{TF0} ) | 0.660        |
| χ²_tot           | **1043.152** |
| χ²_red           | **0.985**    |


🧩 Interpretation

TFGR with mildly phantom-like behavior

​
$$ 
w_{\mathrm{TF}} = -1 - \frac{n_{\mathrm{TF}}}{3} ≈−1.20
 $$
	​

fits the data better than ΛCDM.

This fit becomes the foundation for Phase 187 and 188.

🪐 Phase 187 — Future-Universe Diagnostics

Using the best-fit parameters from Phase 186:

𝑛𝑇𝐹=−0.6

Ω𝑚0=0.34

Ω𝑇𝐹0=0.66

🔥 Key Predictions (TFGR)
| Quantity            | Value                     |
| ------------------- | ------------------------- |
| **j(0)**            | **1.71**                  |
| **j(z = −0.5)**     | **≈ 2.04**                |
| Asymptotic ( j(z) ) | **→ 2.1** as ( z \to -1 ) |
| Acceleration onset  | ( z_{\rm acc} ≈ 0.567 )   |

🌟 Important

In ΛCDM,

𝑗(𝑧)≡1 for all redshifts.

So TFGR predicts a unique dynamical signature that no ΛCDM model can replicate:

increasing jerk in the future,

with present-day value already elevated (j≈1.6–1.7).

This becomes TFGR’s “cosmological fingerprint.”

🧪 Phase 188 — Direct Estimation of j(0) from Data

Using SN+BAO likelihoods and a 2D grid in

(Ω𝑚0,𝑛𝑇𝐹),

Best-fit TFGR (SN+BAO)
| Parameter       | Value     |
| --------------- | --------- |
| ( \Omega_{m0} ) | **0.335** |
| ( n_{TF} )      | **−0.54** |
| ( j(0) )        | **1.636** |


Comparison with ΛCDM
| Model | j(0)      | χ²_tot       |
| ----- | --------- | ------------ |
| ΛCDM  | **1.000** | 1045.648     |
| TFGR  | **1.636** | **1043.137** |

Information Criteria
| Metric | TFGR − ΛCDM                                             |
| ------ | ------------------------------------------------------- |
| Δχ²    | **−2.51** (TFGR better)                                 |
| ΔAIC   | **−0.51** (TFGR better)                                 |
| ΔBIC   | **+4.45** (ΛCDM slightly better due to extra parameter) |

Interpretation

TFGR’s predicted j(0) ≈ 1.6 is statistically consistent with current SN+BAO constraints.

Future precision surveys (Roman, LSST, Euclid) will measure j(0) well enough to decisively test TFGR.

🎯 Scientific Significance
1. TFGR predicts a dynamical jerk: j(z) ≠ 1

This is fundamentally different from ΛCDM and therefore falsifiable.

2. The magnitude j(0)≈1.6 is not arbitrary

It arises naturally from the TFGR power-law time-field scaling.

3. The future prediction j(z→−1)→2.1

Is robust and independent of late-time cosmic variance.
If future data show upward deviation from j=1, TFGR gains very strong support.

4. SN+BAO constraints do NOT rule out TFGR

In fact TFGR slightly outperforms ΛCDM in χ² and AIC.

📌 Conclusion

Phase 186–188 establish jerk-based cosmological diagnostics as one of the most powerful observational tests of TFGR:

TFGR predicts j(0) > 1

ΛCDM predicts j(0)=1 exactly

Data currently allow j(0)≈1.6

Future surveys can confirm or refute TFGR’s prediction

This makes j(z) the most promising direction for near-term experimental falsification or validation of the Time-Field General Relativity framework.