# phase173_tfgr_extra_density_model.py
import numpy as np
import pandas as pd

def omega_extra_powerlaw(z, eps, n, z_star=2.1):
    return eps * ((1.0 + z) / (1.0 + z_star))**n

def H_with_extra(z, H0, Omega_m0, Omega_r0, Omega_t0, eps, n, z_star=2.1):
    z = np.asarray(z)
    # 標準 TFGR/ΛCDM 部分
    E2_base = (Omega_m0 * (1+z)**3
               + Omega_r0 * (1+z)**4
               + Omega_t0)
    # 時間場由来の追加成分
    Omega_extra = omega_extra_powerlaw(z, eps, n, z_star=z_star)
    E2 = E2_base + Omega_extra
    return H0 * np.sqrt(E2), Omega_extra

if __name__ == "__main__":
    # 例: z グリッドと代表パラメータで H(z) を出す
    H0 = 70.0
    Omega_m0 = 0.3
    Omega_r0 = 1.0e-4
    Omega_t0 = 0.7
    eps = 0.05    # ←あとで SN フィットで決める
    n = 3.0       # ←あとで SN フィットで決める

    z_grid = np.linspace(0.0, 1.5, 200)
    H_eff, Omega_extra = H_with_extra(z_grid, H0, Omega_m0, Omega_r0,
                                      Omega_t0, eps, n)
    out = pd.DataFrame({
        "z": z_grid,
        "H_eff_km_s_Mpc": H_eff,
        "Omega_extra": Omega_extra
    })
    out.to_csv("phase173_tfgr_extra_density_model.csv", index=False)
    print("saved phase173_tfgr_extra_density_model.csv")
