# phase42_tfgr_fit.py
# TFGR-only fit to satellite clock bias vs distance L
# Input: phase41_merged_orbit_clock.csv
# Output:
#  - phase42_out/fits/per_sat_params.csv
#  - phase42_out/fits/global_params.txt
#  - phase42_out/merged_with_tfgr.csv
#  - phase42_out/plots/*.png

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# TFGR model
# Δt(L) = dt0 * [1 + (L/Lc)^p]^q
# -----------------------------
def tfgr_dt(L, dt0, Lc, p, q):
    L = np.asarray(L, dtype=float)
    return dt0 * (1.0 + (L / Lc)**p)**q

def safe_curve_fit(x, y, p0, bounds, maxfev=20000):
    """Robust wrapper that returns (params, cov, success_flag)."""
    try:
        popt, pcov = curve_fit(
            tfgr_dt, x, y, p0=p0, bounds=bounds, maxfev=maxfev
        )
        return popt, pcov, True
    except Exception as e:
        return None, None, False

def plot_per_sat(df_sat, popt, out_png, title):
    L = df_sat["L_m"].values
    y = df_sat["clk_bias_s"].values
    yhat = tfgr_dt(L, *popt)

    # dt vs L
    plt.figure()
    plt.scatter(L, y, s=6, label="obs")
    idx = np.argsort(L)
    plt.plot(L[idx], yhat[idx], linewidth=2, label="TFGR fit")
    plt.xlabel("L [m]")
    plt.ylabel("clk_bias_s [s]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_residual(df_sat, popt, out_png, title):
    L = df_sat["L_m"].values
    y = df_sat["clk_bias_s"].values
    yhat = tfgr_dt(L, *popt)
    res = y - yhat

    plt.figure()
    plt.scatter(L, res, s=6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("L [m]")
    plt.ylabel("residual [s]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="phase41_merged_orbit_clock.csv",
                    help="merged orbit+clock csv")
    ap.add_argument("--out_dir", default="phase42_out")
    ap.add_argument("--do_fit", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--nplot_per_fig", type=int, default=10,
                    help="how many satellites per combined figure (unused now but kept for compatibility)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    fits_dir = out_dir / "fits"
    plots_dir = out_dir / "plots"
    fits_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)

    # column sanity
    need_cols = ["time", "sat", "L_m", "clk_bias_s"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {csv_path}")

    # parse time
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["sat", "time"]).reset_index(drop=True)

    # remove invalid
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["L_m", "clk_bias_s"])
    df = df[df["L_m"] > 0]

    sats = sorted(df["sat"].unique())
    print(f"[info] sats={len(sats)} rows={len(df)}")

    # initial guesses / bounds
    # dt0 ~ few 1e-4 to 1e-3 (your CLK biases are around that scale)
    # Lc ~ 4e9 m order (but allow wide)
    p0 = [1e-4, 4e9, 0.2, 1.3]
    bounds = (
        [1e-8, 1e6,  0.01, 0.1],   # lower
        [1e-1, 1e11, 3.0,  5.0]    # upper
    )

    per_sat_rows = []
    df["tfgr_hat_s"] = np.nan
    df["tfgr_res_s"] = np.nan

    # -------- per satellite fit -------
    if args.do_fit:
        print("[fit] per-sat TFGR")
        for sat in sats:
            d = df[df["sat"] == sat]
            x = d["L_m"].values
            y = d["clk_bias_s"].values

            if len(d) < 8:
                continue

            popt, pcov, ok = safe_curve_fit(x, y, p0=p0, bounds=bounds)
            if not ok:
                continue

            dt0_hat, Lc_hat, p_hat, q_hat = popt
            yhat = tfgr_dt(x, *popt)
            res = y - yhat

            rms = np.sqrt(np.mean(res**2))
            per_sat_rows.append({
                "sat": sat,
                "n": len(d),
                "dt0_hat_s": dt0_hat,
                "Lc_hat_m": Lc_hat,
                "p_hat": p_hat,
                "q_hat": q_hat,
                "rms_res_s": rms
            })

            df.loc[d.index, "tfgr_hat_s"] = yhat
            df.loc[d.index, "tfgr_res_s"] = res

            if args.plot:
                plot_per_sat(
                    d.assign(clk_bias_s=y, tfgr_hat_s=yhat),
                    popt,
                    plots_dir / f"{sat}_dt_vs_L.png",
                    f"{sat}: TFGR dt(L) fit"
                )
                plot_residual(
                    d.assign(clk_bias_s=y, tfgr_res_s=res),
                    popt,
                    plots_dir / f"{sat}_res_vs_L.png",
                    f"{sat}: TFGR residual"
                )

        per_sat_df = pd.DataFrame(per_sat_rows).sort_values("sat")
        per_sat_csv = fits_dir / "per_sat_params.csv"
        per_sat_df.to_csv(per_sat_csv, index=False)
        print(f"[saved] per-sat params -> {per_sat_csv}")

        # -------- global fit (all sats pooled) -------
        print("[fit] global TFGR")
        xg = df["L_m"].values
        yg = df["clk_bias_s"].values
        popt_g, pcov_g, ok_g = safe_curve_fit(xg, yg, p0=p0, bounds=bounds)
        if ok_g:
            dt0_g, Lc_g, p_g, q_g = popt_g
            yhat_g = tfgr_dt(xg, *popt_g)
            res_g = yg - yhat_g
            rms_g = np.sqrt(np.mean(res_g**2))

            with open(fits_dir / "global_params.txt", "w", encoding="utf-8") as f:
                f.write("TFGR global fit\n")
                f.write(f"dt0_hat_s = {dt0_g:.6e}\n")
                f.write(f"Lc_hat_m  = {Lc_g:.6e}\n")
                f.write(f"p_hat      = {p_g:.6f}\n")
                f.write(f"q_hat      = {q_g:.6f}\n")
                f.write(f"RMS_res_s  = {rms_g:.6e}\n")

            print(f"[saved] global params -> {fits_dir / 'global_params.txt'}")
        else:
            print("[warn] global fit failed")

    # save merged with residual
    out_csv = out_dir / "merged_with_tfgr.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] saved merged+tfgr -> {out_csv}")

if __name__ == "__main__":
    main()
