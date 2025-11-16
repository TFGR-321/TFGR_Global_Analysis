import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# ==========================================
# Phase 19 — Analytic TFGR Fit (4 parameters)
#   Works for NGC 2903, NGC 5055, etc.
# ==========================================

# ---------- 設定：ここだけ書き換えればOK ----------
GAL_NAME = "NGC 5055"         # "NGC 5055" に変えて再実行すれば5055用
CSV_FILE = "NGC5055.csv"      # 2903用: "NGC2903.csv"
                              # 5055用: "NGC5055.csv"

R_CUT    = 0.5                # フィットに使う最小半径 [kpc]

# 初期値とフィット範囲（2903/5055 共通で使えます）
P0       = (150.0, 10.0, 2.0, 0.7)    # (V0, rc, n, f_disk)
BOUNDS   = ((50.0,  1.0, 0.5, 0.2),   # 下限
            (400.0, 40.0, 8.0, 1.2))  # 上限
# ----------------------------------------------------


# -------------------------------
# 1. データ読み込み
# -------------------------------
df = pd.read_csv(
    CSV_FILE,
    sep=",",
    header=0,
    encoding="utf-8-sig"
).apply(pd.to_numeric, errors="coerce")

df = df.dropna(subset=["Rad", "Vobs"])

r_obs  = df["Rad"].values
Vobs   = df["Vobs"].values
Verr   = df["errV"].values
Vgas   = df["Vgas"].values
Vdisk  = df["Vdisk"].values
Vbul   = df["Vbul"].values

# 誤差が0やNaNの点は仮に 5 km/s を入れる
Verr = np.where((Verr <= 0) | ~np.isfinite(Verr), 5.0, Verr)

# 各成分を補間関数として用意
Vgas_i  = interp1d(r_obs, Vgas, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)
Vdisk_i = interp1d(r_obs, Vdisk, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)
Vbul_i  = interp1d(r_obs, Vbul, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)


# -------------------------------
# 2. TFGR + ディスクスケール付きモデル
# -------------------------------
def v_model_total(r, V0, rc, n, f_disk):
    """
    r      : 半径 [kpc]
    V0     : TFGR 速度振幅 [km/s]
    rc     : TFGR スケール半径 [kpc]
    n      : 立ち上がりの鋭さ
    f_disk : ディスク成分のスケール係数（M/L の自由度）
    """
    r = np.asarray(r)

    # TFGR 成分
    x = np.clip(r / rc, 0.0, 1e3)
    v_tfgr = V0 * np.sqrt(1.0 - np.exp(-x**n))

    # バリオン（ディスクだけ f_disk を掛ける）
    vg = Vgas_i(r)
    vd = Vdisk_i(r) * f_disk
    vb = Vbul_i(r)

    vbar2 = np.maximum(vg**2 + vd**2 + vb**2, 0.0)
    return np.sqrt(np.maximum(vbar2 + v_tfgr**2, 0.0))


# -------------------------------
# 3. パラメータフィット
# -------------------------------
fit_mask = (r_obs > R_CUT)
r_fit    = r_obs[fit_mask]
V_fit    = Vobs[fit_mask]
Verr_fit = Verr[fit_mask]

popt, pcov = curve_fit(
    v_model_total,
    r_fit, V_fit,
    sigma=Verr_fit,
    p0=P0,
    bounds=BOUNDS,
    absolute_sigma=True,
    maxfev=40000
)

V0_best, rc_best, n_best, f_disk_best = popt
perr = np.sqrt(np.diag(pcov))
V0_err, rc_err, n_err, f_disk_err = perr

# -------------------------------
# 4. モデル評価と統計量
# -------------------------------
r_grid = np.linspace(r_obs.min(), r_obs.max(), 500)

# 各成分を分離して描きたいので、もう一度計算
xg = np.clip(r_grid / rc_best, 0.0, 1e3)
Vtfgr_grid = V0_best * np.sqrt(1.0 - np.exp(-xg**n_best))

Vgas_grid  = Vgas_i(r_grid)
Vdisk_grid = Vdisk_i(r_grid) * f_disk_best
Vbul_grid  = Vbul_i(r_grid)
Vbar2_grid = np.maximum(Vgas_grid**2 + Vdisk_grid**2 + Vbul_grid**2, 0.0)
Vbar_grid  = np.sqrt(Vbar2_grid)
Vtot_grid  = np.sqrt(Vbar2_grid + Vtfgr_grid**2)

# フィット統計（観測点位置で）
Vmodel_at_obs = v_model_total(r_obs, V0_best, rc_best, n_best, f_disk_best)
mask_stats = np.isfinite(Vobs) & np.isfinite(Vmodel_at_obs) & (Verr > 0)
rms  = np.sqrt(np.mean((Vobs[mask_stats] - Vmodel_at_obs[mask_stats])**2))
chi2 = np.sum(((Vobs[mask_stats] - Vmodel_at_obs[mask_stats]) /
               Verr[mask_stats])**2) / np.sum(mask_stats)

# -------------------------------
# 5. プロット
# -------------------------------
plt.figure(figsize=(8,6))
plt.errorbar(r_obs, Vobs, yerr=Verr, fmt="o", color="black",
             label=f"Observed ({GAL_NAME})", alpha=0.8)
plt.plot(r_grid, Vbar_grid, "--", color="gray", label="Baryonic (scaled disk)")
plt.plot(r_grid, Vtfgr_grid, ":", color="green", label="TFGR term")
plt.plot(r_grid, Vtot_grid, "-", color="red", linewidth=2.0,
         label="Baryon + TFGR model")

plt.xlabel("Radius r [kpc]")
plt.ylabel("Rotation speed V [km/s]")
plt.title(f"{GAL_NAME} — Analytic TFGR Fit (with disk scaling)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 6. 結果表示
# -------------------------------
print(f"\n--- Analytic TFGR Fit Results ({GAL_NAME}) ---")
print(f"V0      = {V0_best:7.2f} ± {V0_err:6.2f} km/s")
print(f"rc      = {rc_best:7.2f} ± {rc_err:6.2f} kpc")
print(f"n       = {n_best:7.2f} ± {n_err:6.2f}")
print(f"f_disk  = {f_disk_best:7.3f} ± {f_disk_err:6.3f}")
print(f"RMS err = {rms:6.2f} km/s")
print(f"Reduced χ² = {chi2:6.3f}")
print("------------------------------------------------")
