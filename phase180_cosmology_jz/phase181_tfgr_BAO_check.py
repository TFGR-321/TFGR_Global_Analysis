import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- TFGR H(z) model ----------
def H_tfgr(z, H0, Om_m0, Om_SN0, w_eff):
    # SN-mode scaling exponent
    n = 3 * (1 + w_eff)   # ~ -0.24
    return H0 * np.sqrt(Om_m0 * (1 + z)**3 + Om_SN0 * (1 + z)**n)

# ---------- Load BAO/H(z) data (fixed small sample) ----------
def load_bao_data():
    data = {
        "z":    [0.07, 0.17, 0.27, 0.40, 0.57, 0.73],
        "H":    [69.0, 83.0, 77.0, 95.0, 100.3, 97.3],
        "Herr": [19.6, 8.3, 14.0, 17.0, 3.7, 7.0]
    }
    return pd.DataFrame(data)

# ---------- chi2 ----------
def chi2_bao(df, H0, Om_m0, Om_SN0, w_eff):
    chi2 = 0.0
    for i, r in df.iterrows():
        H_mod = H_tfgr(r["z"], H0, Om_m0, Om_SN0, w_eff)
        chi2 += ((H_mod - r["H"])**2) / (r["Herr"]**2)
    return chi2

# ---------- main ----------
def main():
    # Phase 180 TFGR best-fit parameters
    H0 = 70.0
    Om_m0 = 0.30
    Om_SN0 = 0.70
    w_eff = -1.07   # from Phase 178

    df = load_bao_data()
    chi2 = chi2_bao(df, H0, Om_m0, Om_SN0, w_eff)

    print("=== Phase 181: BAO/H(z) check ===")
    print(f"H0       = {H0:.2f}")
    print(f"Omega_m0 = {Om_m0:.3f}")
    print(f"Omega_SN0 = {Om_SN0:.3f}")
    print(f"w_eff    = {w_eff:.3f}")
    print("-----------------------------")
    print(f"chi2_BAO = {chi2:.3f}")
    print("-----------------------------")

    # Plot
    z_plot = np.linspace(0, 1, 200)
    H_plot = H_tfgr(z_plot, H0, Om_m0, Om_SN0, w_eff)

    plt.figure(figsize=(8,6))
    plt.errorbar(df["z"], df["H"], yerr=df["Herr"], fmt="o", label="BAO/H(z) data")
    plt.plot(z_plot, H_plot, label="TFGR model", color="red")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("Phase 181: BAO/H(z) vs TFGR model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("phase181_tfgr_BAO_check.png", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()
