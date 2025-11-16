"""
phase19_tfgr_potential_reconstruct.py

NGC3198 の rotmod データから
  1) TFGR 回転曲線パラメータ (V0, rc, n) をフィット
  2) 時間場ポテンシャル Phi_t(r) を構成
  3) ラプラシアン Box Phi_t(r) = dV/dPhi_t を数値微分で求め
  4) Phi_t を変数として積分し V(Phi_t) を数値的に再構成
するスクリプト。

同じフォルダに NGC3198.csv を置いて実行してください：
    python phase19_tfgr_potential_reconstruct.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

# -----------------------------
# 1. データ読み込み
# -----------------------------
def load_ngc3198(filepath="NGC3198.csv"):
    """
    rotmod 形式の NGC3198 データを読み込む。
    ヘッダー無し CSV（カンマ区切り）を想定。
    列順:
        Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    """
    colnames = ["Rad","Vobs","errV","Vgas","Vdisk","Vbul","SBdisk","SBbul"]
    df = pd.read_csv(filepath, sep=',', header=None, names=colnames)
    # 文字列 → 数値変換（失敗したところは NaN）
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    r = df["Rad"].values
    Vobs = df["Vobs"].values
    Verr = df["errV"].values
    Vgas = df["Vgas"].values
    Vdisk = df["Vdisk"].values
    Vbul = df["Vbul"].values

    Vbar = np.sqrt(Vgas**2 + Vdisk**2 + Vbul**2)

    return r, Vobs, Verr, Vbar

# -----------------------------
# 2. TFGR 回転曲線フィット
# -----------------------------
def fit_tfgr(r, Vobs, Verr, Vbar):
    """
    V_TFGR^2 = V_bar^2 + V0^2 [1 - exp(-(r/rc)^n)]
    を最小二乗でフィットして (V0, rc, n) を返す。
    """
    def Vtfgr_model(r, V0, rc, n):
        return np.sqrt(Vbar**2 + V0**2 * (1.0 - np.exp(-(r/rc)**n)))

    p0 = [120.0, 13.0, 3.0]               # 初期値
    bounds = ([50, 5, 0.5], [300, 30, 8]) # 下限・上限

    popt, pcov = curve_fit(
        Vtfgr_model, r, Vobs, sigma=Verr,
        p0=p0, bounds=bounds, maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))

    V0, rc, n = popt
    print("\n--- TFGR Fit Results (NGC 3198) ---")
    print(f"V0 = {V0:.3f} ± {perr[0]:.3f} km/s")
    print(f"rc = {rc:.3f} ± {perr[1]:.3f} kpc")
    print(f"n  = {n:.3f} ± {perr[2]:.3f}")
    print("-----------------------------------")

    Vmodel = Vtfgr_model(r, *popt)

    return V0, rc, n, Vmodel

# -----------------------------
# 3. Phi_t(r) と Box Phi_t の数値計算
# -----------------------------
def compute_phi_and_laplacian(r, Vmodel, Vbar):
    """
    V_phi^2 = Vmodel^2 - Vbar^2 から Phi_t(r) を定義し、
    数値微分で Box Phi_t = d^2 Phi/dr^2 + (2/r) dPhi/dr を求める。
    """
    c_kms = 3.0e5  # 光速 [km/s]

    # 時間場の寄与速度
    Vphi_sq = np.maximum(Vmodel**2 - Vbar**2, 0.0)
    Vphi = np.sqrt(Vphi_sq)

    # 無次元ポテンシャル Phi_t/c^2 の proxy
    Phi = 0.5 * (Vphi / c_kms)**2   # Phi / c^2

    # 数値微分（非等間隔 r でも np.gradient が対応してくれる）
    dPhi_dr = np.gradient(Phi, r)
    d2Phi_dr2 = np.gradient(dPhi_dr, r)

    # 球対称のラプラシアン
    laplacian = d2Phi_dr2 + 2.0 * dPhi_dr / r  # ≈ dV/dPhi

    return Phi, laplacian

# -----------------------------
# 4. V(Phi) の数値復元
# -----------------------------
def reconstruct_potential(Phi, dVdPhi):
    """
    dV/dPhi(Phi) から V(Phi) を積分で復元。
    Phi を昇順に並べ替え、台形公式で積分する。
    V(Phi_min) = 0 を基準とする。
    """
    # Phi に対して単調になるようソート
    idx = np.argsort(Phi)
    Phi_sorted = Phi[idx]
    dVdPhi_sorted = dVdPhi[idx]

    # 積分：V(Phi) = ∫ (dV/dPhi) dPhi  （台形公式）
    Vphi = cumulative_trapezoid(
        dVdPhi_sorted, Phi_sorted, initial=0.0
    )

    return Phi_sorted, dVdPhi_sorted, Vphi

# -----------------------------
# 5. メイン処理
# -----------------------------
def main():
    # 1) データ読み込み
    r, Vobs, Verr, Vbar = load_ngc3198("NGC3198.csv")

    # 2) TFGR フィット
    V0, rc, n, Vmodel = fit_tfgr(r, Vobs, Verr, Vbar)

    # 3) Phi_t と Box Phi_t
    Phi, laplacian = compute_phi_and_laplacian(r, Vmodel, Vbar)

    # 4) V(Phi) の復元
    Phi_sorted, dVdPhi_sorted, Vphi = reconstruct_potential(Phi, laplacian)

    # 5) 結果を CSV に保存（解析用）
    out = np.column_stack([Phi_sorted, dVdPhi_sorted, Vphi])
    np.savetxt(
        "phase19_tfgr_potential_ngc3198.csv",
        out,
        header="Phi_over_c2,dVdPhi,V_of_Phi (up to const)",
        delimiter=","
    )

    # -----------------------------
    # プロット
    # -----------------------------
    # (a) 回転曲線
    plt.figure(figsize=(8,6))
    plt.errorbar(r, Vobs, yerr=Verr, fmt='o', label="Observed")
    plt.plot(r, Vbar, '--', label="Baryonic")
    plt.plot(r, Vmodel, '-', label="TFGR fit")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title("TFGR Lagrangian Fit: NGC 3198")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (b) Phi_t(r)
    plt.figure(figsize=(7,5))
    plt.plot(r, Phi, '-')
    plt.xlabel("Radius [kpc]")
    plt.ylabel(r"$\Phi_t/c^2$")
    plt.title("Time-Field Potential $\\Phi_t(r)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (c) dV/dPhi vs Phi
    plt.figure(figsize=(7,5))
    plt.plot(Phi_sorted, dVdPhi_sorted, '-')
    plt.xlabel(r"$\Phi_t/c^2$")
    plt.ylabel(r"$\mathrm{d}V/\mathrm{d}\Phi_t$")
    plt.title(r"Effective Force $\,\mathrm{d}V/\mathrm{d}\Phi_t(\Phi_t)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (d) V(Phi)
    plt.figure(figsize=(7,5))
    plt.plot(Phi_sorted, Vphi, '-')
    plt.xlabel(r"$\Phi_t/c^2$")
    plt.ylabel(r"$V(\Phi_t)$ (arb. units)")
    plt.title(r"Reconstructed Potential $V(\Phi_t)$ (up to const.)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
