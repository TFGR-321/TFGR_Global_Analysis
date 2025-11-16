import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

# ======================================================
# Phase 22B — Batch TFGR Analytic Fits (many galaxies)
#   - 入力: SPARC 形式の .csv（Rad, Vobs, errV, Vgas, Vdisk, Vbul, ...）
#   - 出力: 銀河ごとの V0, rc, n, f_disk, RMS, chi2 をまとめた表 + 図
# ======================================================

# 解析したい銀河とファイル名（必要に応じて追加・変更してください）
GALAXIES = [
    ("NGC2403",  "NGC2403.csv"),
    ("NGC3198",  "NGC3198.csv"),
    ("NGC2903",  "NGC2903.csv"),
    ("NGC5055",  "NGC5055.csv"),
    ("DDO154",   "DDO154.csv"),
    ("F571-V1",  "F571-V1.csv"),
    ("IC2574",   "IC2574.csv"),
    ("KK98-251", "KK98-251.csv"),
    ("NGC0055",  "NGC0055.csv"),
    ("NGC0801",  "NGC0801.csv"),
    ("NGC7331",  "NGC7331.csv"),
    ("UGC02259", "UGC02259.csv"),
    ("UGC02953", "UGC02953.csv"),
    ("UGCA281",  "UGCA281.csv"),
    ("F574-1",   "F574-1.csv"),
    ("NGC0024",  "NGC0024.csv"),
    ("NGC0100",  "NGC0100.csv"),
    ("NGC0247",  "NGC0247.csv"),
    ("NGC0289",  "NGC0289.csv"),
    ("NGC0300",  "NGC0300.csv"),
    ("NGC0891",  "NGC0891.csv"),
    ("NGC1003",  "NGC1003.csv"),
    ("NGC1090",  "NGC1090.csv"),
    ("PGC51017", "PGC51017.csv"),
]

# 図の保存先
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# 中心部をフィットから外す半径 [kpc]
R_CUT_DEFAULT = 0.5


def read_rotmod_csv(filename):
    """
    CSV を読み込む補助関数。
    - コメント行 '#' をスキップ
    - 区切り文字を自動判定（カンマ or 空白など）
    - 必要な列を float に変換
    """
    df = pd.read_csv(
        filename,
        comment="#",
        sep=None,          # 区切り自動判定
        engine="python",
        header=0,
        encoding="utf-8-sig"
    )
    # 列名を一応そろえる（大文字小文字ゆらぎ対策）
    df.columns = [c.strip() for c in df.columns]

    # 想定カラム名
    needed = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"{filename}: 列 '{col}' が見つかりません。")

    df = df[needed].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Rad", "Vobs"])
    return df


def tfgr_fit_one_galaxy(name, filename, r_cut=R_CUT_DEFAULT):
    """
    1 銀河分の TFGR + disk scaling フィットを行う。
    返り値: dict（パラメータと統計量）
    """
    print(f"\n=== {name}: reading {filename} ===")

    df = read_rotmod_csv(filename)
    if len(df) == 0:
        print("  -> データ点がありません。スキップします。")
        return {"name": name, "success": False}

    r_obs  = df["Rad"].values
    Vobs   = df["Vobs"].values
    Verr   = df["errV"].values
    Vgas   = df["Vgas"].values
    Vdisk  = df["Vdisk"].values
    Vbul   = df["Vbul"].values

    # 誤差 0 or NaN を仮の値で補う
    Verr = np.where((Verr <= 0) | ~np.isfinite(Verr), 5.0, Verr)

    # 補間関数
    Vgas_i  = interp1d(r_obs, Vgas, kind="cubic",
                       fill_value="extrapolate", bounds_error=False)
    Vdisk_i = interp1d(r_obs, Vdisk, kind="cubic",
                       fill_value="extrapolate", bounds_error=False)
    Vbul_i  = interp1d(r_obs, Vbul, kind="cubic",
                       fill_value="extrapolate", bounds_error=False)

    # --- TFGR + ディスクスケーリングモデル ---
    def v_model_total(r, V0, rc, n, f_disk):
        r = np.asarray(r)
        x = np.clip(r / rc, 0.0, 1e3)
        v_tfgr = V0 * np.sqrt(1.0 - np.exp(-x**n))

        vg = Vgas_i(r)
        vd = Vdisk_i(r) * f_disk
        vb = Vbul_i(r)
        vbar2 = np.maximum(vg**2 + vd**2 + vb**2, 0.0)
        return np.sqrt(np.maximum(vbar2 + v_tfgr**2, 0.0))

    # --- フィットに使う範囲を選択 ---
    r_fit_mask = (r_obs > r_cut)
    if np.sum(r_fit_mask) < 5:
        # データが少なすぎる場合は中央も含める
        r_fit_mask = np.ones_like(r_obs, dtype=bool)

    r_fit    = r_obs[r_fit_mask]
    V_fit    = Vobs[r_fit_mask]
    Verr_fit = Verr[r_fit_mask]

    # --- 銀河ごとの自動初期値 ---
    r_max = np.max(r_fit)
    V_max = np.max(V_fit)

    V0_guess     = max(0.8 * V_max, 50.0)
    rc_guess     = max(0.3 * r_max, 1.0)
    n_guess      = 2.0
    f_disk_guess = 0.7

    p0 = (V0_guess, rc_guess, n_guess, f_disk_guess)

    bounds = (
        (0.3 * V_max, 0.1 * r_max, 0.5, 0.2),   # 下限
        (3.0 * V_max, 3.0 * r_max, 8.0, 1.2),   # 上限
    )

    print(f"  初期値 p0 = {p0}")
    print(f"  フィット点数 N = {len(r_fit)}")

    try:
        popt, pcov = curve_fit(
            v_model_total,
            r_fit, V_fit,
            sigma=Verr_fit,
            p0=p0,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=40000
        )
        V0_best, rc_best, n_best, f_disk_best = popt
        perr = np.sqrt(np.diag(pcov))
        V0_err, rc_err, n_err, f_disk_err = perr
    except Exception as e:
        print(f"  フィット失敗: {e}")
        return {"name": name, "success": False}

    # --- 統計量 ---
    Vmodel_at_obs = v_model_total(r_obs, V0_best, rc_best, n_best, f_disk_best)
    mask_stats = np.isfinite(Vobs) & np.isfinite(Vmodel_at_obs) & (Verr > 0)
    if np.sum(mask_stats) > 4:
        residuals = Vobs[mask_stats] - Vmodel_at_obs[mask_stats]
        rms = np.sqrt(np.mean(residuals**2))
        dof = np.sum(mask_stats) - 4  # パラメータ数4
        chi2_red = np.sum((residuals / Verr[mask_stats])**2) / dof
    else:
        rms = np.nan
        chi2_red = np.nan

    print(f"  => V0={V0_best:.2f}±{V0_err:.2f}, "
          f"rc={rc_best:.2f}±{rc_err:.2f} kpc, "
          f"n={n_best:.2f}±{n_err:.2f}, "
          f"f_disk={f_disk_best:.3f}±{f_disk_err:.3f}")
    print(f"     RMS={rms:.2f} km/s, reduced chi2={chi2_red:.3f}")

    # --- プロット ---
    r_grid = np.linspace(r_obs.min(), r_obs.max(), 500)
    # 成分分離
    xg = np.clip(r_grid / rc_best, 0.0, 1e3)
    Vtfgr_grid = V0_best * np.sqrt(1.0 - np.exp(-xg**n_best))
    Vgas_grid  = Vgas_i(r_grid)
    Vdisk_grid = Vdisk_i(r_grid) * f_disk_best
    Vbul_grid  = Vbul_i(r_grid)
    Vbar2_grid = np.maximum(Vgas_grid**2 + Vdisk_grid**2 + Vbul_grid**2, 0.0)
    Vbar_grid  = np.sqrt(Vbar2_grid)
    Vtot_grid  = np.sqrt(Vbar2_grid + Vtfgr_grid**2)

    plt.figure(figsize=(8,6))
    plt.errorbar(r_obs, Vobs, yerr=Verr, fmt="o", color="black",
                 label=f"Observed ({name})", alpha=0.8)
    plt.plot(r_grid, Vbar_grid, "--", color="gray",
             label="Baryonic (scaled disk)")
    plt.plot(r_grid, Vtfgr_grid, ":", color="green", label="TFGR term")
    plt.plot(r_grid, Vtot_grid, "-", color="red", linewidth=2.0,
             label="Baryon + TFGR model")
    plt.xlabel("Radius r [kpc]")
    plt.ylabel("Rotation speed V [km/s]")
    plt.title(f"{name} — Analytic TFGR Fit (with disk scaling)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(PLOT_DIR, f"{name}_tfgr_fit.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  プロット保存: {out_png}")

    # 結果を dict で返す
    return {
        "name": name,
        "success": True,
        "V0": V0_best,
        "V0_err": V0_err,
        "rc": rc_best,
        "rc_err": rc_err,
        "n": n_best,
        "n_err": n_err,
        "f_disk": f_disk_best,
        "f_disk_err": f_disk_err,
        "RMS": rms,
        "chi2_red": chi2_red,
        "N_points": len(r_fit),
        "r_cut": r_cut,
    }


# ======================================================
# メイン: 全銀河ループ
# ======================================================

results = []

for name, fname in GALAXIES:
    if not os.path.exists(fname):
        print(f"\n*** 注意: ファイル {fname} が見つかりません。スキップします。")
        results.append({"name": name, "success": False})
        continue

    res = tfgr_fit_one_galaxy(name, fname, r_cut=R_CUT_DEFAULT)
    results.append(res)

# pandas DataFrame にまとめて CSV 出力
df_results = pd.DataFrame(results)
df_results.to_csv("tfgr_batch_fit_results.csv", index=False)
print("\n=== 解析完了 ===")
print("結果サマリー: tfgr_batch_fit_results.csv")
print("図:          plots/<Galaxy>_tfgr_fit.png")
