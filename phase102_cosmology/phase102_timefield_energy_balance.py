#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[Phase 102] Time-Field Energy Balance

目的:
    phase100_timefield_energy_flux_profile.csv によるフラックス情報
    phase96_timefield_entropy_profile.csv によるエントロピー情報
    を統合し、

        dF_t/dz + α S_rel,eff(z) = β κ_t φ_t(z)

    の形で α, β を最小二乗で推定する。

出力:
    - <out>_fit_results.txt    : フィット結果のサマリ
    - <out>_fit_profile.csv    : z ごとの結合データと残差など
    - <out>_residual_vs_z.png  : 残差 r(z) の可視化
    - <out>_lhs_rhs_vs_z.png   : LHS(観測)と RHS(モデル) の比較
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# ユーティリティ関数
# -----------------------------
def find_column(df, candidates, description):
    """
    指定された候補リストと部分一致でカラム名を自動検出するヘルパ。

    Parameters
    ----------
    df : pandas.DataFrame
    candidates : list of str
        優先的に探す候補名（完全一致 or 部分一致用キー）
    description : str
        エラー時に表示する説明用の文字列

    Returns
    -------
    col : str
        見つかったカラム名

    Raises
    ------
    ValueError
        見つからなかった場合
    """
    cols = list(df.columns)

    # 1. 完全一致を先に探す
    for cand in candidates:
        if cand in cols:
            return cand

    # 2. 小文字にして部分一致を探す
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        for lc, orig in lower_map.items():
            if key in lc:
                return orig

    raise ValueError(
        f"{description} に対応するカラムが見つかりませんでした。\n"
        f"  候補: {candidates}\n"
        f"  利用可能なカラム: {cols}"
    )


def linear_fit_energy_balance(z, dFdz, S_rel, kappa_phi):
    """
    dF_t/dz + α S_rel = β κ_t φ_t を
    dFdz = β kappa_phi - α S_rel と書き換え、
    最小二乗法で α, β を推定する。

    Parameters
    ----------
    z : array-like
        赤方偏移
    dFdz : array-like
        dF_t/dz
    S_rel : array-like
        S_rel,eff
    kappa_phi : array-like
        κ_t φ_t

    Returns
    -------
    results : dict
        'alpha', 'beta', 'alpha_err', 'beta_err', 'cov', 'resid', 'R2', など。
    """
    z = np.asarray(z)
    y = np.asarray(dFdz)
    S = np.asarray(S_rel)
    K = np.asarray(kappa_phi)

    # 設定したいモデル:
    #   dFdz ≈ β * K - α * S = X θ
    #   ここで θ = [β, α], X = [[K_i, -S_i], ...]
    X = np.vstack([K, -S]).T

    # 最小二乗解
    theta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    beta_hat, alpha_hat = theta[0], theta[1]

    n = len(y)
    p = X.shape[1]

    if residuals.size > 0:
        rss = residuals[0]
    else:
        # データ数とパラメータ数が同じなどで residuals が空の場合は手計算
        y_pred = X @ theta
        rss = np.sum((y - y_pred) ** 2)

    # 分散推定
    if n > p:
        sigma2 = rss / (n - p)
    else:
        # データ不足の場合は単に 0 で埋めておく
        sigma2 = 0.0

    # 共分散行列 (X^T X)^(-1) * sigma^2
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * XtX_inv

    beta_err = float(np.sqrt(cov[0, 0])) if sigma2 > 0 else np.nan
    alpha_err = float(np.sqrt(cov[1, 1])) if sigma2 > 0 else np.nan

    # 予測値 & 残差
    y_pred = X @ theta
    resid = y - y_pred

    # 決定係数 R^2
    tss = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - rss / tss if tss > 0 else np.nan

    results = {
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "alpha_err": alpha_err,
        "beta_err": beta_err,
        "cov": cov,
        "resid": resid,
        "y_pred": y_pred,
        "R2": float(R2),
        "rss": float(rss),
        "sigma2": float(sigma2),
        "n": int(n),
        "p": int(p),
    }
    return results


# -----------------------------
# メイン処理
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="[Phase 102] Time-Field Energy Balance Fitting Script"
    )
    parser.add_argument(
        "--flux_csv",
        required=True,
        help="Phase100 で生成したフラックスプロファイル CSV "
             "(例: phase100_timefield_energy_flux_profile.csv)",
    )
    parser.add_argument(
        "--entropy_csv",
        required=True,
        help="Phase96 で生成したエントロピー関連 CSV "
             "(例: phase96_timefield_entropy_profile.csv)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="出力ファイルのプレフィックス (例: phase102_timefield_energy_balance)",
    )
    parser.add_argument(
        "--z_col_flux",
        default=None,
        help="flux_csv 側の赤方偏移カラム名 (未指定なら自動検出)",
    )
    parser.add_argument(
        "--z_col_entropy",
        default=None,
        help="entropy_csv 側の赤方偏移カラム名 (未指定なら自動検出)",
    )
    parser.add_argument(
        "--dFdz_col",
        default=None,
        help="dF_t/dz カラム名 (未指定なら自動検出)",
    )
    parser.add_argument(
        "--Srel_col",
        default=None,
        help="S_rel,eff カラム名 (未指定なら自動検出)",
    )
    parser.add_argument(
        "--kappa_phi_col",
        default=None,
        help="κ_t φ_t カラム名 (未指定なら自動検出)",
    )
    parser.add_argument(
        "--delta_omega_col",
        default=None,
        help="ΔΩ_Λ,eff カラム名 (任意: 存在すれば相関も計算)",
    )
    parser.add_argument(
        "--merge_tolerance",
        type=float,
        default=1e-4,
        help="z 結合時の許容差 (merge_asof 用, デフォルト: 1e-4)",
    )

    args = parser.parse_args()

    print("[Phase 102] Time-Field Energy Balance")
    print(f"  flux_csv    : {args.flux_csv}")
    print(f"  entropy_csv : {args.entropy_csv}")
    print(f"  out prefix  : {args.out}")
    print("")

    # -----------------------------
    # データ読み込み
    # -----------------------------
    if not os.path.exists(args.flux_csv):
        print(f"[ERROR] flux_csv が見つかりません: {args.flux_csv}")
        sys.exit(1)
    if not os.path.exists(args.entropy_csv):
        print(f"[ERROR] entropy_csv が見つかりません: {args.entropy_csv}")
        sys.exit(1)

    df_flux = pd.read_csv(args.flux_csv)
    df_entropy = pd.read_csv(args.entropy_csv)

    print("[INFO] flux_csv columns   :", list(df_flux.columns))
    print("[INFO] entropy_csv columns:", list(df_entropy.columns))

    # -----------------------------
    # カラムの自動検出
    # -----------------------------
    # z
    if args.z_col_flux is not None:
        z_col_flux = args.z_col_flux
    else:
        z_col_flux = find_column(
            df_flux,
            ["z", "z_bin", "redshift"],
            "flux_csv の赤方偏移 z",
        )

    if args.z_col_entropy is not None:
        z_col_entropy = args.z_col_entropy
    else:
        z_col_entropy = find_column(
            df_entropy,
            ["z", "z_bin", "redshift"],
            "entropy_csv の赤方偏移 z",
        )

    # dF_t/dz
    if args.dFdz_col is not None:
        dFdz_col = args.dFdz_col
    else:
        dFdz_col = find_column(
            df_flux,
            ["dF_t_dz", "dFtdz", "dF_t/dz", "dFdz"],
            "dF_t/dz",
        )

    # κ_t φ_t
    if args.kappa_phi_col is not None:
        kappa_phi_col = args.kappa_phi_col
    else:
        kappa_phi_col = find_column(
            df_flux,
            ["kappa_phi_norm", "kappa_phi", "kappa_t_phi", "kappa_t_phi_norm"],
            "κ_t φ_t",
        )

    # S_rel,eff
    if args.Srel_col is not None:
        Srel_col = args.Srel_col
    else:
        Srel_col = find_column(
            df_entropy,
            ["S_rel_eff", "S_rel", "Srel_eff", "S_rel_norm"],
            "S_rel,eff",
        )

    # ΔΩ_Λ,eff (任意)
    delta_omega_col = None
    if args.delta_omega_col is not None:
        if args.delta_omega_col in df_flux.columns:
            delta_omega_col = args.delta_omega_col
        else:
            print(
                f"[WARN] 指定された ΔΩ カラム {args.delta_omega_col} が flux_csv にありません。"
            )
    else:
        try:
            delta_omega_col = find_column(
                df_flux,
                [
                    "Delta_OmegaL_eff_norm",
                    "DeltaOmegaL_eff_norm",
                    "Delta_OmegaL_eff",
                    "DeltaOmega_eff",
                ],
                "ΔΩ_Λ,eff",
            )
        except ValueError:
            delta_omega_col = None

    print("")
    print("[INFO] 使用カラム名")
    print(f"  z_col_flux     : {z_col_flux}")
    print(f"  z_col_entropy  : {z_col_entropy}")
    print(f"  dFdz_col       : {dFdz_col}")
    print(f"  kappa_phi_col  : {kappa_phi_col}")
    print(f"  Srel_col       : {Srel_col}")
    print(f"  DeltaOmega_col : {delta_omega_col}")
    print("")

    # -----------------------------
    # z での結合 (merge_asof)
    # -----------------------------
    df_flux_sorted = df_flux.sort_values(z_col_flux).reset_index(drop=True)
    df_entropy_sorted = df_entropy.sort_values(z_col_entropy).reset_index(drop=True)

    merged = pd.merge_asof(
        df_flux_sorted,
        df_entropy_sorted[[z_col_entropy, Srel_col]],
        left_on=z_col_flux,
        right_on=z_col_entropy,
        direction="nearest",
        tolerance=args.merge_tolerance,
        suffixes=("_flux", "_entropy"),
    )

    # マージに失敗した行を除く
    merged = merged.dropna(subset=[Srel_col])

    if merged.empty:
        print("[ERROR] merge_asof の結果、結合された行が 0 件でした。")
        print("  merge_tolerance を大きくするか、z のカラム指定を確認してください。")
        sys.exit(1)

    # 分かりやすい列名に揃える
    merged = merged.rename(
        columns={
            z_col_flux: "z",
            dFdz_col: "dF_t_dz",
            kappa_phi_col: "kappa_phi",
            Srel_col: "S_rel_eff",
        }
    )

    # ΔΩ があればリネーム
    if delta_omega_col is not None and delta_omega_col in merged.columns:
        merged = merged.rename(columns={delta_omega_col: "Delta_OmegaL_eff_norm"})

    # 解析に使う列のみ取り出し
    used_cols = ["z", "dF_t_dz", "S_rel_eff", "kappa_phi"]
    if "Delta_OmegaL_eff_norm" in merged.columns:
        used_cols.append("Delta_OmegaL_eff_norm")

    df_used = merged[used_cols].dropna().reset_index(drop=True)

    print(f"[INFO] 結合後のデータ数: {len(df_used)}")
    if len(df_used) < 5:
        print("[WARN] データ点が少ないためフィットの信頼性が低い可能性があります。")

    # -----------------------------
    # フィット実行
    # -----------------------------
    print("")
    print("[INFO] エネルギー収支式フィットを実行します...")
    fit_results = linear_fit_energy_balance(
        df_used["z"].values,
        df_used["dF_t_dz"].values,
        df_used["S_rel_eff"].values,
        df_used["kappa_phi"].values,
    )

    alpha = fit_results["alpha"]
    beta = fit_results["beta"]
    alpha_err = fit_results["alpha_err"]
    beta_err = fit_results["beta_err"]
    R2 = fit_results["R2"]
    resid = fit_results["resid"]
    y_pred = fit_results["y_pred"]
    rss = fit_results["rss"]
    sigma2 = fit_results["sigma2"]
    n = fit_results["n"]

    # 残差と LHS, RHS を df_used に追加
    df_used["energy_balance_LHS"] = df_used["dF_t_dz"] + alpha * df_used["S_rel_eff"]
    df_used["energy_balance_RHS"] = beta * df_used["kappa_phi"]
    df_used["residual_energy_balance"] = df_used["energy_balance_LHS"] - df_used["energy_balance_RHS"]

    # ΔΩ との相関 (存在する場合)
    corr_delta_resid = None
    corr_delta_rhs = None
    if "Delta_OmegaL_eff_norm" in df_used.columns:
        delta = df_used["Delta_OmegaL_eff_norm"].values
        corr_delta_resid = float(np.corrcoef(delta, df_used["residual_energy_balance"].values)[0, 1])
        corr_delta_rhs = float(np.corrcoef(delta, df_used["energy_balance_RHS"].values)[0, 1])

    # -----------------------------
    # 結果の出力
    # -----------------------------
    out_prefix = args.out
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True) if os.path.dirname(out_prefix) else None

    # 1) サマリテキスト
    txt_path = f"{out_prefix}_fit_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Phase 102: Time-Field Energy Balance Fit Results\n\n")
        f.write("Model: dF_t/dz + α S_rel,eff = β κ_t φ_t\n\n")
        f.write(f"Number of data points (n): {n}\n")
        f.write(f"Residual sum of squares (RSS): {rss:.6e}\n")
        f.write(f"Estimated noise variance (sigma^2): {sigma2:.6e}\n")
        f.write(f"R^2: {R2:.6f}\n\n")

        f.write("Fitted parameters:\n")
        f.write(f"  alpha (S_rel,eff coefficient): {alpha:.6e}")
        if not np.isnan(alpha_err):
            f.write(f" ± {alpha_err:.6e}")
        f.write("\n")

        f.write(f"  beta  (kappa_phi coefficient): {beta:.6e}")
        if not np.isnan(beta_err):
            f.write(f" ± {beta_err:.6e}")
        f.write("\n\n")

        if corr_delta_resid is not None:
            f.write("Correlations with Delta_OmegaL_eff_norm:\n")
            f.write(f"  corr(ΔΩ_Λ,eff, residual_energy_balance): {corr_delta_resid:.6f}\n")
            f.write(f"  corr(ΔΩ_Λ,eff, energy_balance_RHS)     : {corr_delta_rhs:.6f}\n")
            f.write("\n")

        f.write("Notes:\n")
        f.write("  - energy_balance_LHS = dF_t/dz + alpha * S_rel,eff\n")
        f.write("  - energy_balance_RHS = beta * kappa_phi\n")
        f.write("  - residual_energy_balance = LHS - RHS\n")

    print(f"[INFO] フィット結果サマリを出力しました: {txt_path}")

    # 2) 結合データ & 残差プロファイル
    csv_path = f"{out_prefix}_fit_profile.csv"
    df_used.to_csv(csv_path, index=False)
    print(f"[INFO] フィットプロファイルを出力しました: {csv_path}")

    # 3) 残差 vs z の図
    fig1_path = f"{out_prefix}_residual_vs_z.png"
    plt.figure()
    plt.axhline(0.0, linestyle="--")
    plt.scatter(df_used["z"], df_used["residual_energy_balance"], s=20)
    plt.xlabel("z")
    plt.ylabel("Residual: LHS - RHS")
    plt.title("Phase 102: Energy Balance Residual vs z")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[INFO] 残差プロットを出力しました: {fig1_path}")

    # 4) LHS, RHS vs z の図
    fig2_path = f"{out_prefix}_lhs_rhs_vs_z.png"
    plt.figure()
    plt.plot(df_used["z"], df_used["energy_balance_LHS"], label="LHS = dF_t/dz + α S_rel,eff")
    plt.plot(df_used["z"], df_used["energy_balance_RHS"], label="RHS = β κ_t φ_t")
    plt.xlabel("z")
    plt.ylabel("Energy balance term")
    plt.title("Phase 102: LHS vs RHS of Energy Balance Equation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[INFO] LHS/RHS 比較プロットを出力しました: {fig2_path}")

    print("")
    print("[Phase 102] 完了: エネルギー収支式のフィット結果が出力されました。")


if __name__ == "__main__":
    main()
