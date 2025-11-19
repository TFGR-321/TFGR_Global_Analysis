import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, os

# ===============================================================
# Time-Field General Relativity (TFGR) GPS Residual Fitting Script
# Author: Takahiro Mitsui
# Version: 2025-11-04
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="TFGR fit on GPS residuals")
    parser.add_argument("--csv", required=True, help="Input CSV file (residuals)")
    parser.add_argument("--out", required=True, help="Output prefix (e.g., output/AJAC)")
    parser.add_argument("--Lc", type=float, default=4.0e9, help="Critical scale Lc [m]")
    parser.add_argument("--p", type=float, default=0.21, help="TFGR exponent p")
    parser.add_argument("--q", type=float, default=1.32, help="TFGR exponent q")
    parser.add_argument("--resid_col", default=None, help="Residual column name (auto-detect if omitted)")
    args = parser.parse_args()

    # ======================
    # 1. Read data
    # ======================
    df = pd.read_csv(args.csv)
    print(f"Loaded: {args.csv} ({len(df)} rows)")

    # --- Auto-detect residual column ---
    possible_cols = ["residual_m", "residual", "residual (m)", "resid", "residuals_m", "Δr_m"]
    if args.resid_col and args.resid_col in df.columns:
        resid_col = args.resid_col
    else:
        resid_col = None
        for c in possible_cols:
            if c in df.columns:
                resid_col = c
                break
        if resid_col is None:
            raise ValueError(f"Residual column not found. Candidates: {possible_cols}")
    print(f"Using residual column: {resid_col}")

    # --- Clean residuals ---
    if "residual_m" in df.columns:
        if df["residual_m"].isna().all() or np.all(df["residual_m"] == 0):
            if "expected_slant_m" in df.columns and df["expected_slant_m"].isna().all():
                # expected_slant_m が空 → 仮の基準線を作る
                base = np.nanmean(df["slant_total_m"])
                df["expected_slant_m"] = base
                print(f"Generated placeholder expected_slant_m = constant({base:.4f})")
            if "expected_slant_m" in df.columns and "slant_total_m" in df.columns:
                df["residual_m"] = df["slant_total_m"] - df["expected_slant_m"]
                print("Generated residual_m = slant_total_m - expected_slant_m.")
            else:
                print("Warning: expected_slant_m not found. residual_m left empty.")
    df[resid_col] = pd.to_numeric(df[resid_col], errors="coerce")
    df = df.dropna(subset=[resid_col])
    print(f"Valid data points: {len(df)}")

    # ======================
    # 2. Range (L) estimation
    # ======================
    if "slant_total_m" in df.columns:
        df["range_m"] = df["slant_total_m"]
        print("Using slant_total_m as range_m.")
    elif "elev_deg" in df.columns:
        Re, h = 6.371e6, 2.02e7
        elev_rad = np.radians(df["elev_deg"].clip(lower=3.0))  # lower bound 3 deg
        df["range_m"] = (Re * h) / (Re + h) * 1/np.sin(elev_rad)
        print("Computed range_m from elevation angle.")
    else:
        raise ValueError("No range_m or elev_deg found. Distance info required.")

    # ======================
    # 3. TFGR correction term
    # ======================
    L, Lc, p, q = df["range_m"].values, args.Lc, args.p, args.q
    phi_t = (1 + (L / Lc)**p)**q - 1  # temporal field term
    r = df[resid_col].values

    # ======================
    # 4. Linear fit (residuals vs. Φ_t)
    # ======================
    mask = np.isfinite(phi_t) & np.isfinite(r)
    if np.sum(mask) < 3:
        raise ValueError("Not enough valid data points for fitting.")

    X = np.vstack([phi_t[mask], np.ones_like(phi_t[mask])]).T
    y = r[mask]
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, b = coeff

    resid_after = y - (A*phi_t[mask] + b)

    # ======================
    # 5. Statistics
    # ======================
    def rms(x): return np.sqrt(np.nanmean((x - np.nanmean(x))**2))
    n = len(y)
    rss0 = np.nansum((y - np.nanmean(y))**2)
    rss1 = np.nansum((resid_after - np.nanmean(resid_after))**2)
    rms0, rms1 = rms(y), rms(resid_after)
    AIC0 = n*np.log(rss0/n)
    AIC1 = n*np.log(rss1/n) + 2*2
    dAIC = AIC0 - AIC1

    # ======================
    # 6. Save results
    # ======================
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_csv = f"{args.out}_with_model.csv"
    out_txt = f"{args.out}_results.txt"
    df_out = df.copy()
    df_out["phi_t"] = phi_t
    df_out["fit_residual"] = np.nan
    df_out.loc[mask, "fit_residual"] = resid_after
    df_out.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"TFGR Fit Results for {args.csv}\n")
        f.write("="*60 + "\n")
        f.write(f"Data points: {n}\n")
        f.write(f"A (scale coefficient): {A:.3e}\n")
        f.write(f"b (offset): {b:.3e}\n")
        f.write(f"RMS_before: {rms0:.3e}\n")
        f.write(f"RMS_after:  {rms1:.3e}\n")
        f.write(f"ΔRMS: {rms1-rms0:.3e}\n")
        f.write(f"AIC_before: {AIC0:.2f}\n")
        f.write(f"AIC_after:  {AIC1:.2f}\n")
        f.write(f"ΔAIC (improvement): {dAIC:.2f}\n")
        f.write("="*60 + "\n")
    print(f"Saved results: {out_txt}")

    # ======================
    # 7. Visualization
    # ======================
    plt.figure(figsize=(6,5))
    plt.scatter(phi_t, r, s=8, alpha=0.6, label="Observed residuals")
    xline = np.linspace(np.nanmin(phi_t), np.nanmax(phi_t), 200)
    plt.plot(xline, A*xline + b, "r--", lw=2, label="TFGR fit")
    plt.xlabel("Φₜ(L)")
    plt.ylabel("Residual [m]")
    plt.legend()
    plt.title("TFGR Fit on Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out}_scatter_fit.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(y, bins=40, alpha=0.6, label="Before")
    plt.hist(resid_after, bins=40, alpha=0.6, label="After")
    plt.xlabel("Residual [m]")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Residual Distribution Before/After TFGR")
    plt.tight_layout()
    plt.savefig(f"{args.out}_hist.png", dpi=200)
    plt.close()

    print("✅ TFGR fitting completed successfully.")

# ===============================================================
if __name__ == "__main__":
    main()
