#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase51_gps_tfgr_tomography.py

Phase 51: 地球–衛星–月系における時間場 Φ_t(L) の垂直トモグラフィー解析用スクリプト（初期版）

想定する入力:
    事前に整形した CSV ファイル（例: gps_phase51_example.csv）

必須列:
    - time        : 観測時刻（任意のフォーマット。ここでは解析には使わない）
    - sat         : 衛星 ID 文字列 (例: G01, G02, ...)
    - L_m         : 地球中心から衛星までの距離 L [m] もしくは高度
    - dt_res_s    : 「GR + 標準補正式」まで入れた後の時計残差 Δt_res [秒]
任意列:
    - dt_err_s    : 残差の 1σ 不確かさ [秒]（あれば χ² 評価に使用）

機能:
    1. CSV の読み込み
    2. TFGR 時間補正関数 Δt_TFGR(L) の形状 f(L; Lc, p, q) を計算
    3. 「GR-only（定数オフセットのみ）」モデルと
       「GR + TFGR（定数オフセット + f(L)）」モデルを最小二乗でフィット
    4. 各モデルの χ², AIC, BIC を計算
    5. Δt_res vs L の散布図と、TFGR フィット曲線を重ねた図を出力
    6. L–Δt–Φ_t の簡易 3D サーフェス / 散布図を出力（オプション）

使い方の例:
    python phase51_gps_tfgr_tomography.py \\
        --csv gps_phase51_example.csv \\
        --out phase51_test \\
        --Lc 4.0e9 --p 0.21 --q 1.32

"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


C_LIGHT = 299792458.0  # m/s, 真空中の光速


def tfgr_shape(L, Lc, p, q):
    """
    TFGR の時間補正関数 Δt(L) = Δt0 [1 + (L/Lc)^p]^q の
    「形状部分」 f(L) = [1 + (L/Lc)^p]^q - 1 を返す。

    実際のモデルは
        Δt_model(L) = A * f(L) + B
    としてフィットする（A は有効な Δt0 に比例、B は定数オフセット）。
    """
    L = np.asarray(L, dtype=float)
    x = (L / Lc) ** p
    return (1.0 + x) ** q - 1.0


def fit_linear_model(x_design, y, yerr=None):
    """
    一般化線形モデル y = X beta + noise を最小二乗でフィットする簡易関数。
    X: (N, k) デザイン行列
    y: (N,) 観測値
    yerr: (N,) 誤差 (オプション)。あれば重み付き最小二乗。

    戻り値:
        beta_hat      : (k,) 推定パラメータ
        y_model       : (N,) 予測値
        chi2          : χ²
        dof           : 自由度
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(x_design, dtype=float)

    if yerr is not None:
        w = 1.0 / np.asarray(yerr, dtype=float) ** 2
        # W^(1/2) を掛けて通常の最小二乗に帰着
        W_sqrt = np.sqrt(w)
        Xw = X * W_sqrt[:, None]
        yw = y * W_sqrt
        beta_hat, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        y_model = X @ beta_hat
        chi2 = np.sum(((y - y_model) / yerr) ** 2)
    else:
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_model = X @ beta_hat
        # 誤差が無い場合は単純に平方和を χ² 相当として扱う
        chi2 = np.sum((y - y_model) ** 2)

    dof = max(len(y) - X.shape[1], 1)
    return beta_hat, y_model, chi2, dof


def compute_ic(chi2, n_params, n_data):
    """
    AIC, BIC を計算する。
    log-likelihood ~ -0.5 * chi2 とみなした簡易形。
    """
    aic = chi2 + 2 * n_params
    bic = chi2 + n_params * np.log(max(n_data, 1))
    return aic, bic


def make_plots(df, Lc, p, q, beta_tfgr, out_prefix):
    """
    残差 vs L プロットと、L–Δt–Φ_t の簡易 3D プロットを作成。
    """
    L = df["L_m"].values
    dt_res = df["dt_res_s"].values

    # TFGR 形状とモデル
    fL = tfgr_shape(L, Lc=Lc, p=p, q=q)
    A, B = beta_tfgr
    dt_model = A * fL + B

    # 1. Δt_res vs L (log10 L) + モデル曲線
    fig, ax = plt.subplots()
    ax.scatter(L, dt_res, s=5, alpha=0.5, label="Residuals (GR-corrected)")
    # モデル曲線用にソート
    idx = np.argsort(L)
    ax.plot(L[idx], dt_model[idx], linewidth=2.0, label="GR + TFGR fit")

    ax.set_xscale("log")
    ax.set_xlabel("L [m] (geocentric distance or altitude)")
    ax.set_ylabel("Clock residual Δt [s]")
    ax.set_title("Phase 51: GPS TFGR tomography (Δt vs L)")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_dt_vs_L.png", dpi=300)
    plt.close(fig)

    # 2. L–Δt–Φ_t 3D 散布図（簡易版）
    # Φ_t(L) = c^2 * Δt(L) / Δt0 だが、ここでは正規化が任意なので、
    # 便宜上 Δt をそのまま使った「相対量」として扱う。
    # （本格版では Δt0 の物理的定義に合わせて再実装する）
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")

    # φ軸として「モデル Φ_t(L)」に相当する量を一応プロット
    # Φ_t(L) ∝ Δt_model とみなし、スケーリング定数は無視
    Phi_rel = dt_model * C_LIGHT**2

    sc = ax3d.scatter(
        L, dt_res, Phi_rel, s=5, c=np.log10(L), cmap=cm.viridis, alpha=0.7
    )
    ax3d.set_xlabel("L [m]")
    ax3d.set_ylabel("Δt_res [s]")
    ax3d.set_zlabel("Φ_t (relative units)")
    ax3d.set_title("Phase 51: Vertical time-field tomography (L–Δt–Φ_t)")

    fig.colorbar(sc, ax=ax3d, label="log10(L/m)")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_L_dt_Phi3D.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 51: GPS/LLR-based vertical tomography of the time field Φ_t(L) (TFGR)"
    )
    parser.add_argument("--csv", required=True, help="Input CSV file with L_m, dt_res_s, etc.")
    parser.add_argument("--out", required=True, help="Output prefix for figures / CSV.")
    parser.add_argument("--Lc", type=float, default=4.0e9, help="Critical length scale Lc [m].")
    parser.add_argument("--p", type=float, default=0.21, help="Nonlinear index p.")
    parser.add_argument("--q", type=float, default=1.32, help="Nonlinear index q.")
    parser.add_argument(
        "--alpha", type=float, default=3.0, help="Growth index alpha (for info only)."
    )
    parser.add_argument(
        "--min_L", type=float, default=None, help="Optional lower cut for L [m]."
    )
    parser.add_argument(
        "--max_L", type=float, default=None, help="Optional upper cut for L [m]."
    )
    parser.add_argument(
        "--use_error",
        action="store_true",
        help="Use dt_err_s column as 1σ error for weighted least squares if present.",
    )

    args = parser.parse_args()

    csv_path = args.csv
    out_prefix = args.out

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["L_m"]
    if "dt_res_s" not in df.columns and "dt_est_s" in df.columns:
        df["dt_res_s"] = df["dt_est_s"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in CSV.")

    # L の範囲でフィルタ
    if args.min_L is not None:
        df = df[df["L_m"] >= args.min_L]
    if args.max_L is not None:
        df = df[df["L_m"] <= args.max_L]

    df = df.dropna(subset=["L_m", "dt_res_s"]).copy()
    n_data = len(df)
    if n_data < 2:
        raise RuntimeError(f"Too few data points after filtering: N={n_data}")


    L = df["L_m"].values
    dt_res = df["dt_res_s"].values

    # 必要なら誤差も読む
    yerr = None
    if args.use_error and "dt_err_s" in df.columns:
        yerr = df["dt_err_s"].values
    elif args.use_error:
        print("Warning: use_error is True but 'dt_err_s' column not found. Using unweighted fit.")

    # --- モデル 1: GR-only (定数オフセット) ---
    X_gr = np.ones((n_data, 1))  # 列ベクトル [1]
    beta_gr, y_gr, chi2_gr, dof_gr = fit_linear_model(X_gr, dt_res, yerr=yerr)
    aic_gr, bic_gr = compute_ic(chi2_gr, n_params=1, n_data=n_data)

    # --- モデル 2: GR + TFGR (定数 + f(L)) ---
    fL = tfgr_shape(L, Lc=args.Lc, p=args.p, q=args.q)
    X_tfgr = np.column_stack([fL, np.ones_like(fL)])  # [f(L), 1]
    beta_tfgr, y_tfgr, chi2_tfgr, dof_tfgr = fit_linear_model(X_tfgr, dt_res, yerr=yerr)
    aic_tfgr, bic_tfgr = compute_ic(chi2_tfgr, n_params=2, n_data=n_data)

    # ΔAIC, ΔBIC (TFGR - GR)。負の値なら TFGR が優位。
    delta_aic = aic_tfgr - aic_gr
    delta_bic = bic_tfgr - bic_gr

    # 結果のサマリを表示
    print("=== Phase 51: TFGR tomography fit summary ===")
    print(f"N data          : {n_data}")
    print(f"Lc [m]         : {args.Lc:.3e}")
    print(f"p, q, alpha    : {args.p:.3f}, {args.q:.3f}, {args.alpha:.3f}")
    print("")
    print("Model GR-only (constant offset):")
    print(f"  beta_gr (offset) = {beta_gr[0]:.6e} [s]")
    print(f"  chi2, dof        = {chi2_gr:.3f}, {dof_gr}")
    print(f"  AIC, BIC         = {aic_gr:.3f}, {bic_gr:.3f}")
    print("")
    print("Model GR + TFGR (A * f(L) + B):")
    print(f"  beta_tfgr (A,B)  = {beta_tfgr[0]:.6e}, {beta_tfgr[1]:.6e}")
    print(f"  chi2, dof        = {chi2_tfgr:.3f}, {dof_tfgr}")
    print(f"  AIC, BIC         = {aic_tfgr:.3f}, {bic_tfgr:.3f}")
    print("")
    print("Information criteria difference (TFGR - GR):")
    print(f"  ΔAIC = {delta_aic:.3f}  (negative favours TFGR)")
    print(f"  ΔBIC = {delta_bic:.3f}  (negative favours TFGR)")

    # 結果を CSV に出力
    summary = {
        "N_data": n_data,
        "Lc_m": args.Lc,
        "p": args.p,
        "q": args.q,
        "alpha": args.alpha,
        "beta_gr_offset_s": beta_gr[0],
        "chi2_gr": chi2_gr,
        "dof_gr": dof_gr,
        "AIC_gr": aic_gr,
        "BIC_gr": bic_gr,
        "beta_tfgr_A_s": beta_tfgr[0],
        "beta_tfgr_B_s": beta_tfgr[1],
        "chi2_tfgr": chi2_tfgr,
        "dof_tfgr": dof_tfgr,
        "AIC_tfgr": aic_tfgr,
        "BIC_tfgr": bic_tfgr,
        "delta_AIC_tfgr_minus_gr": delta_aic,
        "delta_BIC_tfgr_minus_gr": delta_bic,
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = f"{out_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")

    # 図の作成
    make_plots(df, Lc=args.Lc, p=args.p, q=args.q, beta_tfgr=beta_tfgr, out_prefix=out_prefix)
    print(f"Figures saved with prefix: {out_prefix}_*.png")


if __name__ == "__main__":
    main()
