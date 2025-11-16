
# Phase 29 — GPS×TFGR Residual Fitter

Adds a Time-Field (TFGR) correction to post-GR GPS clock residuals and evaluates improvement.

## Files
- `phase29_gps_tfgr_fit.py` : main fitter
- `gps_example_template.csv` : minimal input template

## Minimal CSV
time_utc,sat,elev_deg,residual_s,sigma_s,range_m
2025-11-03 12:00:00,G01,45.0,2.5e-12,1.0e-12,2.70e7
2025-11-03 12:00:30,G01,50.0,2.1e-12,1.0e-12,2.68e7
2025-11-03 12:01:00,G01,55.0,1.9e-12,1.0e-12,2.66e7

## Example
python phase29_gps_tfgr_fit.py --csv gps_example_template.csv --out test_run   --Lc 4.0e9 --p 0.21 --q 1.32 --Lmode elev --fit_mode A
