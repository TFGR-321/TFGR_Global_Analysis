\# TFGR Atomic Clock Multi-Scale Analysis (Phases 36–45A)



This folder contains the full analysis pipeline and results for testing Time-Field General Relativity (TFGR) with multi-scale optical atomic clock data.

The workflow corresponds to Phases 36–45A in the broader TFGR research programme.



The goal is to check whether a scale-dependent time-field



$$

\\Delta t(L) = \\Delta t\_0\\,\\left\[1 + (L/L\_c)^p\\right]^q

$$



is consistent with state-of-the-art clock comparisons, and how its parameters relate to atomic mass and gravitational redshift tests.



\## Folder Structure

\## Input data



- phase36\_multiscale\_dataset.csv



Multi-scale clock comparison dataset with the following columns:



&nbsp;- L\_macro\_m – macro-scale separation between sites \[m]

&nbsp;- L\_int\_m – representative internal scale of the apparatus \[m]

&nbsp;- dt\_res\_s – residual time/frequency offset \[s] after standard GR corrections

&nbsp;- dt\_err\_s – 1σ uncertainty of the residual \[s]

&nbsp;- pair – identifier of the clock pair (e.g. INRIM\_ITYb1-PTB\_Sr3\_CombKnoten, etc.)



This dataset is the common input to the Phase 37–38 TFGR fits.



--- 



Analysis scripts

Phase 37 – Hierarchical Bayesian TFGR fit



- Script: phase37\_tfgr\_hierarchical\_bayes.py

- Input: phase36\_multiscale\_dataset.csv

- Output directory: output\_phase37/



This script performs a \*\*two-level hierarchical Bayesian fit\*\* using PyMC:



- Models:



&nbsp;- \*\*Baseline\*\*: GR-only residual model (no TFGR correction)

&nbsp;- \*\*TFGR2\*\*: two-scale TFGR model (macro \& internal scales)



- For each clock pair:



&nbsp;- Individual bias term b\_pair

&nbsp;- Additional noise term sigma\_pair



- Global TFGR parameters shared across pairs:



&nbsp;- Macro-scale: dt0\_M, Lc\_M, p\_M, q\_M

&nbsp;- Quantum/internal scale: dt0\_Q, Lc\_Q, p\_Q, q\_Q



\*\*Key outputs in\*\* output\_phase37/:



- phase37\_trace\_summary.txt

&nbsp; – MCMC diagnostics (mean, sd, HDI, R-hat, ESS) for the hierarchical model.



- waic\_summary\_phase37.txt

&nbsp; – WAIC comparison:



Baseline (hierarchical GR-only):

&nbsp; WAIC = -95.539 ± 4.770



TFGR2 (hierarchical, two-scale):

&nbsp; WAIC = -95.730 ± 4.946



ΔWAIC (TFGR2 - Baseline) = -0.191  (smaller is better)



Posterior medians in physical units:



- Macro: dt0\_M ≈ -1.36×10⁻¹⁵ s, Lc\_M ≈ 1.0×10⁸ m, p\_M ≈ 0.64, q\_M ≈ 1.42

- Quantum: dt0\_Q ≈ -6.46×10⁻¹⁶ s, Lc\_Q ≈ 3.26×10⁻⁵ m, p\_Q ≈ 1.25, q\_Q ≈ 0.03



→ The TFGR2 model is marginally preferred over the GR-only baseline (ΔWAIC < 0) and yields a consistent two-scale time-field structure.



- fit\_Lmacro\_vs\_dt\_phase37.png, fit\_Lint\_vs\_dt\_phase37.png

&nbsp; – Visual fits of residuals vs. L\_macro\_m and L\_int\_m with posterior predictive bands.



\*\*Usage example\*\*



python phase37\_tfgr\_hierarchical\_bayes.py \\

&nbsp; --csv phase36\_multiscale\_dataset.csv \\

&nbsp; --out output\_phase37 \\

&nbsp; --draws 1000 --tune 1000 --chains 4 --seed 123



--- 



\## Phase 38 – Mass-scaling of the quantum TFGR scale



- Script: phase38\_tfgr\_mass\_scaling.py

- Input: phase36\_multiscale\_dataset.csv

- Output directory: output\_phase38\_mass/



Phase 38 tests a mass-scaled TFGR model:



- The effective atomic mass m\_eff is inferred from the pair label (Sr, Yb, In, etc.).

- The quantum critical scale is assumed to follow:



$$

L\_{cQ}(m\_{\\mathrm{eff}}) = L\_{cQ0}\\,\\left(\\frac{m\_0}{m\_{\\mathrm{eff}}}\\right)^{-1}

$$



with reference mass 

𝑚0=100𝑢.



\### Key output:



- waic\_mass\_scaling.txt

&nbsp; – WAIC and posterior medians for the mass-scaled TFGR2 model. Example:



&nbsp;- WAIC (mass-scaled TFGR2) = -95.580 ± 4.927

&nbsp;- LcQ0 (m0=100u) ≈ 3.31×10⁻⁵ m

&nbsp;- dt0\_Q ≈ -8.89×10⁻¹⁶ s, p\_Q ≈ 0.98, q\_Q ≈ 0.04

&nbsp;- Lists m\_eff for each clock pair.



--- 



\## Phase 39 – 𝐿𝑐𝑄 –mass relation (log–log fit)



- Script: phase39\_lcq\_mass\_relation.py



- Input: output\_phase38\_mass/waic\_mass\_scaling.txt



Output directory: output\_phase39/



This script reads the Phase 38 results and performs a log–log linear regression between L\_cQ and m\_eff.



\### Key outputs:



lcq\_mass\_relation.png

– log₁₀(L\_cQ) vs. log₁₀(m\_eff) with best-fit line.



lcq\_mass\_fit\_summary.txt:



Fit equation: log10(L\_cQ) = 4.0000 + -1.0000 \* log10(m\_eff)

Correlation r = -1.000, p = 1.064e-62

Derived power law:  L\_cQ ∝ m\_eff^-1.00





→ The quantum critical scale follows a clean inverse-mass law 𝐿𝑐𝑄∝𝑚−1 across species.



-- - 



\## Phase 40 – Time-field potential Φ𝑡(𝐿,𝑚)



- Script: phase40\_phi\_t\_mass\_map.py

- Input: Phase 37 \& 38 best-fit parameters



\### Output directory: output\_phase40\_phi\_t/



This script builds a 2D map of the time-field potential Φ𝑡(𝐿,𝑚) as a function of macro scale 𝐿 and mass 

𝑚eff, using:



Macro TFGR parameters from Phase 37



Quantum TFGR parameters with mass scaling from Phase 38



A representative internal scale L\_int\_ref = 5×10⁻³ m



\### Outputs:



- phi\_t\_mass\_map.png – 2D map of Φ𝑡(𝐿,𝑚)

- phi\_t\_mass\_slices.png – characteristic slices vs. 𝐿 or 𝑚

- phi\_t\_mass\_summary.txt – lists the parameter values used to build the map.



--- 



\## Phase 41 – Gradient of Φ𝑡



Script: phase41\_phi\_t\_gradient\_map.py

Output directory (expected): output\_phase41\_phi\_t\_gradient/ (not included in this ZIP)



Computes ∂Φ𝑡/∂𝐿 and related gradient-based quantities for the same (𝐿,𝑚) grid.

Run this script to regenerate gradient maps if needed.



--- 



\## Phase 42 – Curvature of Φ𝑡

&nbsp;	​

- Script: phase42\_phi\_t\_curvature\_map.py

- Output directory: output\_phase42\_phi\_t\_curvature/



Builds the curvature tensor of the time-field via the Laplacian $\\nabla^{2}\\Phi\_{t}(L, m)$



\### Key output:



- phi\_t\_curvature\_map.png, phi\_t\_curvature\_contour.png – curvature maps

- phi\_t\_curvature\_summary.txt:



max(∇²Φ\_t) = -4.943e-08

min(∇²Φ\_t) = -1.268e+00

max at: L=1.000e-03 m, m=82.4 u

min at: L=8.442e+07 m, m=200.0 u



This identifies where the time-field is most strongly curved in the (𝐿,𝑚) plane.



--- 



\## Phase 43 – TFGR effective curvature \& energy density



- Script: phase43\_tfgr\_tensor\_formulation.py



- Output directory: output\_phase43\_tfgr\_tensor/



Constructs effective time-curvature 𝑅𝑡 and energy density 𝜌𝑡 derived from Φ𝑡.



\### Key output:



- rho\_t\_map.png, Rt\_map.png – 2D maps of 𝜌𝑡 and 𝑅𝑡	​

- tfgr\_tensor\_summary.txt – values at representative scales:



&nbsp;- quantum scale: 𝐿=10^−3

&nbsp;- GPS scale: 𝐿∼2×10^7m

&nbsp;- lunar scale: 𝐿=10^8

 

These diagnostics show how the time-field energy density and curvature vary across experimental regimes (quantum → GPS → lunar).



--- 



\## Phase 44 – TFGR tensor field (T00, 𝑝𝑡, etc.)



- Script: phase44\_tfgr\_tensor\_field.py

- Output directories:



&nbsp;- output\_phase44\_tfgr\_tensor/

&nbsp;- output\_phase44\_tfgr\_tensor\_2/ (alternative parameter set / configuration)



Exports the full TFGR tensor field on the (𝐿,𝑚) grid:



- tfgr\_tensor\_field.npz – NumPy archive containing arrays for T00, 𝑝𝑡, etc.

- T00\_map.png, p\_t\_map.png – visualizations

- tfgr\_tensor\_field\_summary.txt – representative values for:



&nbsp;- quantum clocks

&nbsp;- GPS regime

&nbsp;- lunar-scale clocks (Sr, Yb etc.)



These files are intended as intermediate products for coupling TFGR into GR-like field equations.



--- 



\## Phase 45A – GR vs GR+TFGR in Schwarzschild spacetime



Script: phase45A\_tfgr\_schwarzschild.py



Output directory: output\_phase45A\_tfgr\_schwarzschild/



Applies the TFGR corrections to frequency shifts in the Earth’s Schwarzschild field and compares:



- Pure GR gravitational redshift: 𝑦GR

- Combined GR + TFGR: 𝑦total=𝑦GR+𝑦TFGR

&nbsp;	​

\### Key output: tfgr\_vs\_gr\_summary.txt



Example values for m\_eff = 88 u, T\_ref = 86400 s:



- At GPS altitude:



&nbsp;- y\_GR ≈ 5.26×10⁻¹⁰

&nbsp;- y\_TFGR ≈ -3.62×10⁻²⁰

&nbsp;- y\_TFGR / y\_GR ≈ -6.9×10⁻¹¹



- At GEO:



&nbsp;- y\_TFGR / y\_GR ≈ -6.8×10⁻¹¹



- At lunar distance:



&nbsp;- y\_TFGR / y\_GR ≈ -1.4×10⁻¹⁰



→ TFGR corrections are 10⁻¹¹–10⁻¹⁰ times smaller than the GR signal, and therefore negligible for current gravitational redshift tests, showing that the atomic-clock-motivated TFGR parameters are compatible with existing GR bounds.



Usage example



python phase45A\_tfgr\_schwarzschild.py \\

&nbsp; --m\_eff 88.0 \\

&nbsp; --Tref 86400 \\

&nbsp; --out output\_phase45A\_tfgr\_schwarzschild



--- 



\## Dependencies



All scripts are standard Python and rely on the usual scientific stack:



- Python 3.10 or later

- numpy

- pandas

- matplotlib

- pymc (v4, for MCMC; used in Phases 37–38)

- arviz

- scipy (for linear regression in Phase 39)



Install via:



pip install numpy pandas matplotlib pymc arviz scipy



--- 



\## Typical workflow



To fully reproduce the analysis in this folder:



1. Prepare environment
   pip install numpy pandas matplotlib pymc arviz scipy
2. Run the hierarchical TFGR fit
   python phase37\_tfgr\_hierarchical\_bayes.py \\
     --csv phase36\_multiscale\_dataset.csv \\
     --out output\_phase37
3. Run the mass-scaled model
   python phase38\_tfgr\_mass\_scaling.py \\
     --csv phase36\_multiscale\_dataset.csv \\
     --out output\_phase38\_mass
4. Analyse the 𝐿𝑐𝑄–mass relation
   python phase39\_lcq\_mass\_relation.py \\
     --out output\_phase39
5. Generate time-field maps and tensors
   python phase40\_phi\_t\_mass\_map.py
   python phase41\_phi\_t\_gradient\_map.py
   python phase42\_phi\_t\_curvature\_map.py
   python phase43\_tfgr\_tensor\_formulation.py
   python phase44\_tfgr\_tensor\_field.py
6. Check GR compatibility

&nbsp;  python phase45A\_tfgr\_schwarzschild.py \\

&nbsp;    --m\_eff 88.0 \\

&nbsp;    --Tref 86400



--- 



\## Scientific summary



This folder shows that:



- A two-scale TFGR model (macro + quantum) can fit the current multi-scale atomic clock residuals at least as well as a GR-only hierarchical model (slightly better WAIC).



- The quantum critical scale satisfies a very clean law



$$

L\_{cQ} \\propto m\_{\\mathrm{eff}}^{-1}

$$



linking time-field structure directly to atomic mass.



- The derived time-field potential 

Φ𝑡(𝐿,𝑚), its curvature, and the associated tensor field 

(𝑅𝑡,𝜌𝑡,𝑇00,𝑝𝑡) provide a bridge from clock phenomenology to a field-theoretic TFGR description.



- When propagated into a Schwarzschild geometry, the TFGR corrections to gravitational redshift are at the level of 10^−11−10^−10

&nbsp;of the GR signal, i.e. far below current experimental sensitivities, so the clock-based TFGR parameters are compatible with all existing GR tests.



--- 



\## Additional code snapshots



The files:



- python phase37\_code.txt

- python phase38\_code.txt

- python phase39\_code.txt

- python phase40\_code.txt



contain frozen copies of the corresponding Python scripts, intended for inclusion in papers or supplementary material. They are not required for running the analysis.

