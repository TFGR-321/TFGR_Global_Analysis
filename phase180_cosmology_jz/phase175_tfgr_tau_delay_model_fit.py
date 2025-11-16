#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 175 : TFGR time-field delay model fit to SN residuals

目的
-----
TFGR の基本距離 μ_TFGR(z) に対する SN 距離のズレ

    Δμ_data(z) = μ_SN(z) − μ_TFGR(z) − C_best

を、時間場 Φ_t の「緩和時間 τ(z)」に由来する効果として

    Δμ_model(z) = A * z^1.5 * (1+z)^m

でフィットする。

ここで
  A  ~  (係数 k) × τ0
  m  ~  τ(z) の赤方偏移依存 τ(z) = τ0 (1+z)^m

に対応する。

実行例
-------
python phase175_tfgr_tau_delay_model_fit.py \
    --sn_csv pantheon_SN.csv \
    --tfgr_csv phase161_tfgr_plateau_H0_70_Hz_qz_Omegaz_mu.csv \
    --z_min 0.01 --z_max 1.5 \
    --m_min -2.0 --m_max 4.0 --m_steps 600 \
    --out_prefix phase175_tfgr_tau_delay
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sn(sn_csv, z_col="z", mu_col="mu", mu_err_col="mu_err"):
    """Pantheon SN データを読み込む"""
    df = pd.read_csv(sn_csv)
    for c in [z_col, mu_col, mu_err_col]:
        if c not in df.columns:
            raise RuntimeError(f"SN CSV に列 {c} がありません: {df.columns}")
    z = df[z_col].values.astype(float)
    mu = df[mu_col].values.astype(float)
    mu_err = df[mu_err_col].values.astype(float)
    return z, mu, mu_err


def load_tfgr(tfgr_csv):
    """TFGR 距離テーブル (phase161…) を読み込む"""
    df = pd.read_csv(tfgr_csv)
    required = ["z", "mu"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"TFGR CSV に列 {c} がありません: {df.columns}")
    z = df["z"].values.astype(float)
    mu = df["mu"].values.astype(float)
    return z, mu


def interpolate_mu_tfgr(z_sn, z_tfgr, mu_tfgr):
    """TFGR の μ(z) を SN の z に線形補間"""
    # z 範囲外は NaN にする
    mu_interp = np.interp(z_sn, z_tfgr, mu_tfgr,
                          left=np.nan, right=np.nan)
    mask = np.isfinite(mu_interp)
    return mu_interp, mask


def fit_for_m(m, z, y, y_err):
    """
    固定した m に対して
        y(z) ≃ C + A * z^1.5 * (1+z)^m
    を誤差つき最小二乗でフィットする。

    y = μ_SN − μ_TFGR
    """
    f = (z ** 1.5) * ((1.0 + z) ** m)

    # 重み w = 1/σ^2
    w = 1.0 / (y_err ** 2)

    S   = np.sum(w)
    S_f = np.sum(w * f)
    S_ff = np.sum(w * f * f)
    S_y = np.sum(w * y)
    S_fy = np.sum(w * f * y)

    # 2×2 正規方程式を解析的に解く
    Delta = S * S_ff - S_f ** 2
    if Delta == 0.0:
        return np.nan, np.nan, np.inf

    C = (S_ff * S_y - S_f * S_fy) / Delta
    A = (S * S_fy - S_f * S_y) / Delta

    model = C + A * f
    chi2 = np.sum(w * (y - model) ** 2)

    return C, A, chi2


def main():
    parser = argparse.ArgumentParser(
        description="Phase175: TFGR tau-delay model fit to SN data"
    )
    parser.add_argument("--sn_csv", required=True,
                        help="Pantheon SN CSV (z, mu, mu_err)")
    parser.add_argument("--tfgr_csv", required=True,
                        help="TFGR distance CSV (phase161...)")
    parser.add_argument("--z_min", type=float, default=0.0,
                        help="使用する SN の最小 z (default=0.0)")
    parser.add_argument("--z_max", type=float, default=1.5,
                        help="使用する SN の最大 z (default=1.5)")
    parser.add_argument("--m_min", type=float, default=-2.0,
                        help="m グリッドの最小値 (default=-2.0)")
    parser.add_argument("--m_max", type=float, default=4.0,
                        help="m グリッドの最大値 (default=4.0)")
    parser.add_argument("--m_steps", type=int, default=600,
                        help="m グリッドの分割数 (default=600)")
    parser.add_argument("--out_prefix", required=True,
                        help="出力ファイルのプレフィックス")
    args = parser.parse_args()

    print("=== Phase 175: TFGR tau-delay model fit ===")
    print(f"SN CSV   : {args.sn_csv}")
    print(f"TFGR CSV : {args.tfgr_csv}")
    print(f"z range  : [{args.z_min:.3f}, {args.z_max:.3f}]")
    print(f"m range  : [{args.m_min:.3f}, {args.m_max:.3f}] "
          f"({args.m_steps} steps)")
    print(f"out_prefix: {args.out_prefix}")
    print("==========================================")

    # 1. データ読み込み
    z_sn, mu_sn, mu_sn_err = load_sn(args.sn_csv)
    z_tfgr, mu_tfgr = load_tfgr(args.tfgr_csv)

    # 2. TFGR μ(z) を補間
    mu_tfgr_interp, mask_interp = interpolate_mu_tfgr(
        z_sn, z_tfgr, mu_tfgr
    )

    # 3. z 範囲と補間有効範囲でマスク
    mask = (
        mask_interp &
        (z_sn >= args.z_min) &
        (z_sn <= args.z_max)
    )
    z = z_sn[mask]
    mu_data = mu_sn[mask]
    mu_err = mu_sn_err[mask]
    mu_tfgr_used = mu_tfgr_interp[mask]

    N = len(z)
    if N == 0:
        raise RuntimeError("有効な SN データ点が 0 です。z_min/z_max や TFGR 範囲を確認してください。")

    print(f"[Info] SN points used = {N}")

    # 4. y = μ_SN − μ_TFGR を作る
    y = mu_data - mu_tfgr_used

    # 5. m グリッドを走査して (C, A) をフィット
    m_grid = np.linspace(args.m_min, args.m_max, args.m_steps)
    chi2_list = []
    C_list = []
    A_list = []

    print("[Info] scanning m-grid...")
    for m in m_grid:
        C, A, chi2 = fit_for_m(m, z, y, mu_err)
        chi2_list.append(chi2)
        C_list.append(C)
        A_list.append(A)

    chi2_arr = np.array(chi2_list)
    C_arr = np.array(C_list)
    A_arr = np.array(A_list)

    # 6. ベスト m を選ぶ
    idx_best = np.argmin(chi2_arr)
    m_best = m_grid[idx_best]
    C_best = C_arr[idx_best]
    A_best = A_arr[idx_best]
    chi2_min = chi2_arr[idx_best]
    dof = N - 2  # パラメータ：A, C (m はグリッド固定扱い)
    chi2_red = chi2_min / dof

    print("------ Best-fit tau-delay model ------")
    print(f"m_best   = {m_best:.5f}")
    print(f"A_best   = {A_best:.5f}")
    print(f"C_best   = {C_best:.5f} mag")
    print(f"chi2_min = {chi2_min:.3f}")
    print(f"dof      = {dof}")
    print(f"chi2_red = {chi2_red:.3f}")
    print("--------------------------------------")

    # 7. ベストフィットの Δμ_model(z) と Δμ_data(z) を計算
    f_best = (z ** 1.5) * ((1.0 + z) ** m_best)
    delta_mu_model = A_best * f_best
    # 観測側 Δμ_data = μ_SN − μ_TFGR − C_best
    delta_mu_data = y - C_best

    # 8. CSV 出力
    out_csv = args.out_prefix + "_delta_mu_fit.csv"
    df_out = pd.DataFrame({
        "z": z,
        "mu_SN": mu_data,
        "mu_err": mu_err,
        "mu_TFGR": mu_tfgr_used,
        "delta_mu_data": delta_mu_data,
        "delta_mu_model": delta_mu_model
    })
    df_out.to_csv(out_csv, index=False)
    print(f"[Output] Δμ fit table -> {out_csv}")

    # 9. サマリー CSV
    out_sum = args.out_prefix + "_summary.csv"
    df_sum = pd.DataFrame({
        "m_best": [m_best],
        "A_best": [A_best],
        "C_best": [C_best],
        "chi2_min": [chi2_min],
        "dof": [dof],
        "chi2_red": [chi2_red],
        "z_min_used": [args.z_min],
        "z_max_used": [args.z_max],
        "N_SN_used": [N]
    })
    df_sum.to_csv(out_sum, index=False)
    print(f"[Output] summary -> {out_sum}")

    # 10. Δμ(z) プロット
    plt.figure(figsize=(9, 6))
    plt.scatter(z, delta_mu_data, s=8, alpha=0.4, label="SN residuals")
    z_plot = np.linspace(z.min(), z.max(), 400)
    f_plot = (z_plot ** 1.5) * ((1.0 + z_plot) ** m_best)
    delta_mu_plot = A_best * f_plot
    plt.plot(z_plot, delta_mu_plot, "r", lw=2,
             label=r"best-fit $\Delta\mu(z)$")
    plt.axhline(0.0, color="k", ls="--", lw=1)
    plt.xlabel("z")
    plt.ylabel(r"$\Delta\mu(z) = \mu_{\rm SN} - \mu_{\rm TFGR} - C_{\rm best}$")
    plt.title("Phase 175: SN residuals and best-fit tau-delay model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = args.out_prefix + "_delta_mu_fit.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Output] plot -> {out_png}")

    # 11. χ^2(m) プロット（おまけ）
    plt.figure(figsize=(8, 5))
    plt.plot(m_grid, chi2_arr / dof, marker="o", ms=3)
    plt.axvline(m_best, color="r", ls="--",
                label=fr"best m = {m_best:.3f}")
    plt.xlabel("m (tau(z) ~ (1+z)^m)")
    plt.ylabel(r"$\chi^2_{\rm red}$")
    plt.title("Phase 175: chi^2_red vs m (tau-delay model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_chi = args.out_prefix + "_chi2_vs_m.png"
    plt.tight_layout()
    plt.savefig(out_chi, dpi=150)
    plt.close()
    print(f"[Output] chi2(m) plot -> {out_chi}")

    print("=== Phase 175 finished ===")


if __name__ == "__main__":
    main()
