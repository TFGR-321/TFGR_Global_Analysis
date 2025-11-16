# phase174_tfgr_extra_density_SN_fit.py
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

C_LIGHT = 299792.458  # km/s

def H_with_extra(z, H0, Omega_m0, Omega_r0, Omega_t0, eps, n, z_star=2.1):
    z = np.asarray(z)
    E2_base = (Omega_m0 * (1+z)**3
               + Omega_r0 * (1+z)**4
               + Omega_t0)
    Omega_extra = eps * ((1.0 + z) / (1.0 + z_star))**n
    E2 = E2_base + Omega_extra
    return H0 * np.sqrt(E2)

def mu_model(z, H0, Omega_m0, Omega_r0, Omega_t0, eps, n):
    z = np.asarray(z)
    # H(z)
    H = H_with_extra(z, H0, Omega_m0, Omega_r0, Omega_t0, eps, n)
    # 積分用の細かいグリッド
    z_int = np.linspace(0.0, z.max(), 2000)
    H_int = H_with_extra(z_int, H0, Omega_m0, Omega_r0, Omega_t0, eps, n)
    invH = 1.0 / H_int
    D_C_int = cumulative_trapezoid(invH, z_int, initial=0.0) * C_LIGHT  # [Mpc]
    # 補間して D_C(z) を取得
    D_C = np.interp(z, z_int, D_C_int)
    D_L = (1.0 + z) * D_C
    mu = 5.0 * np.log10(D_L) + 25.0
    return mu

def chi2_eps_n(params, z_sn, mu_sn, mu_err,
               H0, Omega_m0, Omega_r0, Omega_t0):
    eps, n = params
    mu_th = mu_model(z_sn, H0, Omega_m0, Omega_r0, Omega_t0, eps, n)
    w = 1.0 / (mu_err**2)
    # C_best を解析的に求める
    C_best = np.sum(w * (mu_sn - mu_th)) / np.sum(w)
    residuals = mu_sn - (mu_th + C_best)
    chi2 = np.sum(w * residuals**2)
    return chi2

if __name__ == "__main__":
    # SN データ
    sn = pd.read_csv("pantheon_SN.csv")
    z_sn = sn["z"].values
    mu_sn = sn["mu"].values
    mu_err = sn["mu_err"].values

    # 宇宙論パラメータ
    H0 = 70.0
    Omega_m0 = 0.3
    Omega_r0 = 1.0e-4
    Omega_t0 = 0.7  # TFGR の plateau からの値で OK

    # 初期値（Phase172 の感触を反映して少し大きめ）
    eps_init = 0.05
    n_init = 3.0

    res = minimize(
        chi2_eps_n,
        x0=np.array([eps_init, n_init]),
        args=(z_sn, mu_sn, mu_err, H0, Omega_m0, Omega_r0, Omega_t0),
        method="Nelder-Mead"
    )

    eps_best, n_best = res.x
    chi2_min = res.fun
    dof = len(z_sn) - 2 - 1  # eps, n, C の3パラ
    chi2_red = chi2_min / dof

    # ベストフィットの C_best と残差も保存
    mu_th_best = mu_model(z_sn, H0, Omega_m0, Omega_r0, Omega_t0,
                          eps_best, n_best)
    w = 1.0 / (mu_err**2)
    C_best = np.sum(w * (mu_sn - mu_th_best)) / np.sum(w)
    residuals = mu_sn - (mu_th_best + C_best)

    summary = pd.DataFrame([{
        "eps_best": eps_best,
        "n_best": n_best,
        "C_best": C_best,
        "chi2_min": chi2_min,
        "dof": dof,
        "chi2_red": chi2_red
    }])
    summary.to_csv("phase174_tfgr_extra_density_SNfit_summary.csv",
                   index=False)

    res_df = pd.DataFrame({
        "z": z_sn,
        "mu_SN": mu_sn,
        "mu_err": mu_err,
        "mu_model": mu_th_best + C_best,
        "residual": residuals
    })
    res_df.to_csv("phase174_tfgr_extra_density_SNfit_residuals.csv",
                  index=False)

    print("=== Phase 174: TFGR extra-density SN fit ===")
    print(summary.to_string(index=False))
