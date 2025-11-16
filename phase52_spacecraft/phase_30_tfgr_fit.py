#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 30: Extraterrestrial TFGR Verification and Time-Field Clock Design
Real-data fitting script (GR vs TFGR, with L_c, p, q fixed by default).

Usage examples:
  python phase30_tfgr_fit.py --csv rosetta_mag_example.tab --fmt rosetta_mag --units_km \
      --dtcol dt_residual --out output/rosetta_test --fit_mode fixed

  python phase30_tfgr_fit.py --simulate --out output/sim_test
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -------------------------------------------------------------------------
# TFGR correction function
#   Δt(L) = Δt0 * [1 + (L/Lc)**p]**q
# -------------------------------------------------------------------------

def tfgr_dt(L, dt0, Lc, p, q):
    L = np.asarray(L, dtype=float)
    return dt0 * (1.0 + (L / Lc) ** p) ** q


# -------------------------------------------------------------------------
# GR baseline model: here we take "no scale dependence", i.e. constant offset
# -------------------------------------------------------------------------

def gr_dt(L, dt_const):
    return np.full_like(np.asarray(L, dtype=float), dt_const)


# -------------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------------

def load_generic_csv(path, dtcol, x_col="X", y_col="Y", z_col="Z", units_km=False):
    """
    Generic loader: expects columns time, X, Y, Z and dtcol.
    Units can be in km (units_km=True) or m (default).
    """
    df = pd.read_csv(path, comment="#", sep=r"\s+|,|\t", engine="python")
    required = {x_col, y_col, z_col, dtcol}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # rename for internal consistency
    df = df.rename(columns={x_col: "X", y_col: "Y", z_col: "Z", dtcol: "dt_res"})
    if units_km:
        df[["X", "Y", "Z"]] *= 1e3  # km -> m

    df["L"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2)
    return df


def load_rosetta_mag(path, dtcol=None, units_km=True):
    """
    Very simple Rosetta MAG-like loader, based on example snippet:
    UTC, ID, X, Y, Z, Bx, By, Bz, ... (comma-separated)
    There is no official dt_residual column in the raw data, so by default
    this function will NOT create dt_res; the user should have precomputed
    it and added as an extra column (name given by dtcol).
    """
    df = pd.read_csv(path, comment="#", header=None)
    # heuristic: first 5 columns = time, id, X, Y, Z
    if df.shape[1] < 5:
        raise ValueError("Rosetta MAG file appears to have <5 columns; loader may need adjustment.")

    df = df.rename(columns={0: "time", 1: "ID", 2: "X", 3: "Y", 4: "Z"})
    if units_km:
        df[["X", "Y", "Z"]] *= 1e3

    if dtcol is not None:
        # dtcol is an index or column name; support both
        if isinstance(dtcol, int):
            df = df.rename(columns={dtcol: "dt_res"})
        else:
            df = df.rename(columns={dtcol: "dt_res"})
    elif "dt_res" not in df.columns:
        raise ValueError(
            "No dt_res column found and dtcol not provided. "
            "Please add a residual column to the MAG file first."
        )

    df["L"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2)
    return df


def load_rpcica_tab(path, dtcol=None, units_km=True):
    """
    Simple loader for RPC-ICA L4-like TAB data, based on example snippet:
      2016-09-03T00:02:28.075,0, 14.9, 0.1, 0.4, ... (elevation table, etc.)
    Often these files do not contain spacecraft position; for TFGR you will
    typically need an external ephemeris to compute L(t). Here we assume that
    the user has already merged such information and added columns X,Y,Z and
    dt_res (or another column name given by dtcol).
    """
    df = pd.read_csv(path, comment="#", sep=r"\s+|,", engine="python", header=None)
    # For safety, we do not enforce a specific column layout here.
    if dtcol is not None:
        if isinstance(dtcol, int):
            df = df.rename(columns={dtcol: "dt_res"})
        else:
            df = df.rename(columns={dtcol: "dt_res"})
    if "X" not in df.columns or "Y" not in df.columns or "Z" not in df.columns:
        raise ValueError(
            "RPC-ICA file does not contain X,Y,Z columns. "
            "Please pre-merge trajectory information and add these columns."
        )
    if units_km:
        df[["X", "Y", "Z"]] *= 1e3

    if "dt_res" not in df.columns:
        raise ValueError("RPC-ICA file has no dt_res column; please provide dtcol or add dt_res.")

    df["L"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2)
    return df


# -------------------------------------------------------------------------
# Fitting and statistics
# -------------------------------------------------------------------------

def compute_aic(residuals, k):
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    rss = np.sum(residuals ** 2)
    if rss <= 0 or n <= 0:
        return np.nan
    return n * np.log(rss / n) + 2 * k


def fit_gr(L, dt_obs):
    """Fit GR baseline (constant residual)."""
    dt_obs = np.asarray(dt_obs, dtype=float)
    dt_const = np.mean(dt_obs)
    dt_model = gr_dt(L, dt_const)
    res = dt_obs - dt_model
    aic = compute_aic(res, k=1)
    rms = np.sqrt(np.mean(res ** 2))
    return dt_const, dt_model, res, aic, rms


def fit_tfgr_fixed(L, dt_obs, Lc=4.0e9, p=0.21, q=1.32, dt0_init=1e-12):
    """
    Fit only Δt0 with (Lc, p, q) fixed from previous phases.
    """
    L = np.asarray(L, dtype=float)
    dt_obs = np.asarray(dt_obs, dtype=float)

    def model(L_val, dt0):
        return tfgr_dt(L_val, dt0, Lc=Lc, p=p, q=q)

    popt, pcov = curve_fit(model, L, dt_obs, p0=[dt0_init], maxfev=10000)
    dt0_fit = popt[0]
    dt_model = model(L, dt0_fit)
    res = dt_obs - dt_model
    aic = compute_aic(res, k=1)  # only dt0
    rms = np.sqrt(np.mean(res ** 2))
    dt0_err = float(np.sqrt(pcov[0, 0])) if pcov.size > 0 else np.nan
    return dt0_fit, dt0_err, dt_model, res, aic, rms


def fit_tfgr_free(L, dt_obs, Lc_init=4.0e9, p_init=0.21, q_init=1.32, dt0_init=1e-12):
    """
    Fit all four parameters (Δt0, Lc, p, q) freely.
    Use with care: parameters are often highly correlated.
    """
    L = np.asarray(L, dtype=float)
    dt_obs = np.asarray(dt_obs, dtype=float)

    def model(L_val, dt0, Lc, p, q):
        return tfgr_dt(L_val, dt0, Lc, p, q)

    popt, pcov = curve_fit(
        model,
        L,
        dt_obs,
        p0=[dt0_init, Lc_init, p_init, q_init],
        maxfev=20000,
    )
    dt_model = model(L, *popt)
    res = dt_obs - dt_model
    aic = compute_aic(res, k=4)
    rms = np.sqrt(np.mean(res ** 2))
    perr = np.sqrt(np.diag(pcov)) if pcov.size == 16 else np.full(4, np.nan)
    return popt, perr, dt_model, res, aic, rms


# -------------------------------------------------------------------------
# Optical link baseline calibration for Δt0
# -------------------------------------------------------------------------

def calibrate_dt0_from_optical_link(dt0_est, fractional_instability=7e-17):
    """
    Very rough Δt0 calibration using a 2220 km optical fibre link with
    fractional frequency instability ~7×10⁻17 (Nature Comms 2022).
    """
    c = 299_792_458.0  # m/s
    reference_path = 2.22e6  # 2220 km
    delta_t_opt = fractional_instability * reference_path / c
    # Scale dt0_est so that 1e-12 corresponds to delta_t_opt
    scale = delta_t_opt / 1e-12
    return dt0_est * scale, delta_t_opt


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_fit(L, dt_obs, dt_gr, dt_tfgr, out_png):
    plt.figure(figsize=(8, 5))
    plt.scatter(L, dt_obs, s=8, alpha=0.6, label="Observed residuals")
    plt.plot(L, dt_tfgr, lw=2, label="TFGR fit", color="red")
    plt.plot(L, dt_gr, lw=1.5, ls="--", label="GR baseline", color="gray")
    plt.xscale("log")
    plt.xlabel("Distance L [m]")
    plt.ylabel("Δt residual [s]")
    plt.title("TFGR vs GR beyond Critical Length Scale")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# Simulation helper (for動作確認)
# -------------------------------------------------------------------------

def simulate_dataset(n_points=300, L_min=1e7, L_max=1e12,
                     dt0=1e-12, Lc=4e9, p=0.21, q=1.32,
                     noise_sigma=2e-13):
    L = np.logspace(np.log10(L_min), np.log10(L_max), n_points)
    dt_true = tfgr_dt(L, dt0, Lc, p, q)
    dt_obs = dt_true + np.random.normal(0.0, noise_sigma, size=L.size)
    df = pd.DataFrame({"L": L, "dt_res": dt_obs})
    return df


# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 30: GR vs TFGR fit beyond critical length scale."
    )
    parser.add_argument("--csv", type=str, help="Input data file (.csv/.tab/.txt).")
    parser.add_argument("--fmt", type=str, default="generic",
                        choices=["generic", "rosetta_mag", "rpcica"],
                        help="Input file format.")
    parser.add_argument("--units_km", action="store_true",
                        help="Treat X,Y,Z as km and convert to m.")
    parser.add_argument("--dtcol", type=str,
                        help="Column name (or integer index for some loaders) of residual Δt [s].")
    parser.add_argument("--xcol", type=str, default="X", help="X column name (generic only).")
    parser.add_argument("--ycol", type=str, default="Y", help="Y column name (generic only).")
    parser.add_argument("--zcol", type=str, default="Z", help="Z column name (generic only).")
    parser.add_argument("--out", type=str, default="phase30_output",
                        help="Output prefix (without extension).")
    parser.add_argument("--fit_mode", type=str, default="fixed",
                        choices=["fixed", "free"],
                        help="TFGR fit mode: 'fixed' (Δt0 only) or 'free' (Δt0,Lc,p,q).")
    parser.add_argument("--simulate", action="store_true",
                        help="Use synthetic dataset instead of reading file.")
    args = parser.parse_args()

    if not args.simulate and not args.csv:
        parser.error("Either --simulate or --csv must be specified.")

    # ------------------------------------------------------------------
    # Load or simulate data
    # ------------------------------------------------------------------
    if args.simulate:
        print(">> Using synthetic TFGR dataset for test.")
        df = simulate_dataset()
    else:
        path = args.csv
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if args.fmt == "generic":
            df = load_generic_csv(path, dtcol=args.dtcol or "dt_res",
                                  x_col=args.xcol, y_col=args.ycol, z_col=args.zcol,
                                  units_km=args.units_km)
            print("=== DEBUG: Columns ===")
            print(df.columns.tolist())
            print(df.head(5))
            print(df["dt_res"].describe())

        elif args.fmt == "rosetta_mag":
            # dtcol can be name or integer; if None, expect existing 'dt_res'
            try:
                dtcol_val = int(args.dtcol) if args.dtcol is not None and args.dtcol.isdigit() else args.dtcol
            except AttributeError:
                dtcol_val = args.dtcol
            df = load_rosetta_mag(path, dtcol=dtcol_val, units_km=args.units_km)
        elif args.fmt == "rpcica":
            try:
                dtcol_val = int(args.dtcol) if args.dtcol is not None and args.dtcol.isdigit() else args.dtcol
            except AttributeError:
                dtcol_val = args.dtcol
            df = load_rpcica_tab(path, dtcol=dtcol_val, units_km=args.units_km)
        else:
            raise ValueError(f"Unsupported fmt={args.fmt}")

    # Ensure L and dt_res columns exist
    if "L" not in df.columns or "dt_res" not in df.columns:
        raise ValueError("Dataframe must contain columns 'L' and 'dt_res'.")

    # Sort by distance (for nicer plots)
    df = df.sort_values("L")
    L = df["L"].values
    dt_obs = df["dt_res"].values

    # ------------------------------------------------------------------
    # Fit GR baseline
    # ------------------------------------------------------------------
    dt_const, dt_gr, res_gr, aic_gr, rms_gr = fit_gr(L, dt_obs)

    # ------------------------------------------------------------------
    # Fit TFGR
    # ------------------------------------------------------------------
    if args.fit_mode == "fixed":
        dt0_fit, dt0_err, dt_tfgr, res_tfgr, aic_tfgr, rms_tfgr = fit_tfgr_fixed(L, dt_obs)
        print("\n=== TFGR (fixed Lc,p,q) fit ===")
        print(f"Δt0 = {dt0_fit:.3e} ± {dt0_err:.3e} s")
        print("L_c, p, q are fixed to: 4.0e9 m, 0.21, 1.32")
    else:
        popt, perr, dt_tfgr, res_tfgr, aic_tfgr, rms_tfgr = fit_tfgr_free(L, dt_obs)
        print("\n=== TFGR (free) fit ===")
        print(f"Δt0 = {popt[0]:.3e} ± {perr[0]:.3e} s")
        print(f"L_c = {popt[1]:.3e} ± {perr[1]:.3e} m")
        print(f"p   = {popt[2]:.3f} ± {perr[2]:.3f}")
        print(f"q   = {popt[3]:.3f} ± {perr[3]:.3f}")

    print("\n=== Model comparison (GR vs TFGR) ===")
    print(f"GR:    AIC = {aic_gr:.3f}, RMS = {rms_gr:.3e} s")
    print(f"TFGR:  AIC = {aic_tfgr:.3f}, RMS = {rms_tfgr:.3e} s")
    print(f"ΔAIC (GR - TFGR) = {aic_gr - aic_tfgr:.3f}")

    # ------------------------------------------------------------------
    # Δt0 calibration using optical link baseline
    # ------------------------------------------------------------------
    if args.fit_mode == "fixed":
        dt0_est = dt0_fit
    else:
        dt0_est = popt[0]

    dt0_cal, delta_t_opt = calibrate_dt0_from_optical_link(dt0_est)
    print("\n=== Δt0 calibration (optical link baseline) ===")
    print(f"Optical link Δt (2220 km, 7e-17) ≈ {delta_t_opt:.3e} s")
    print(f"Calibrated Δt0 ≈ {dt0_cal:.3e} s")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_png = args.out + "_tfgr_fit.png"
    plot_fit(L, dt_obs, dt_gr, dt_tfgr, out_png)
    print(f"\nFigure saved to: {out_png}\n")


if __name__ == "__main__":
    main()
