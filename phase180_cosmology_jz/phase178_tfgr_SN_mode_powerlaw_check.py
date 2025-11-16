#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 178B: TFGR SN-mode power-law check

目的:
  ソース付き連続の式
    dρ_SN/dz - [3(1+w_*)/(1+z)] ρ_SN = - S0 (1+z)^s
  を解析的に解いた形を使って ρ_SN(z) を計算し、
  ρ_SN(z) ~ (1+z)^n のべき則でどの程度よく近似できるかを確認する。

やること:
  1. パラメータ (w_star, s, rho0, S0) を指定して ρ_SN(z) を計算
  2. ログ空間で log10 ρ_SN vs log10(1+z) を線形フィットして n を取得
  3. 結果を表示し、CSV とプロット(PNG)を保存

依存:
  numpy, pandas, matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_rho_SN(z, w_star=-1.0, s=0.5, rho0=1.0, S0=0.1):
    """
    ソース付き連続式の解析解に基づいて ρ_SN(z) を計算する。

    dρ/dz - [3(1+w*)/(1+z)] ρ = - S0 (1+z)^s

    積分因子 I(z) = (1+z)^(-3(1+w*))
    => d/dz [ρ I] = - S0 (1+z)^s I
                  = - S0 (1+z)^{s - 3(1+w*)}
    """
    z = np.asarray(z)
    I = (1.0 + z)**(-3.0 * (1.0 + w_star))

    exp_int = s - 3.0 * (1.0 + w_star)
    tol = 1e-8

    if abs(exp_int + 1.0) > tol:
        # ∫ (1+z')^exp_int dz' = ((1+z)^{exp_int+1} - 1)/(exp_int+1)
        integral = ((1.0 + z)**(exp_int + 1.0) - 1.0) / (exp_int + 1.0)
    else:
        # exp_int ≈ -1 の場合は ln(1+z)
        integral = np.log(1.0 + z)

    rho_SN = (1.0 + z)**(3.0 * (1.0 + w_star)) * (rho0 - S0 * integral)
    return rho_SN


def fit_power_law(z, rho, z_min_fit=0.01, z_max_fit=1.5):
    """
    ρ(z) を ρ(z) ≈ A (1+z)^n でフィットする。
    ログ空間で線形フィット: log10 ρ = n log10(1+z) + log10 A
    """
    z = np.asarray(z)
    rho = np.asarray(rho)

    mask = (z >= z_min_fit) & (z <= z_max_fit) & (rho > 0.0)
    z_fit = z[mask]
    rho_fit = rho[mask]

    if len(z_fit) < 3:
        raise RuntimeError("フィットに使えるデータ点が少なすぎます。パラメータや z 範囲を見直してください。")

    x = np.log10(1.0 + z_fit)
    y = np.log10(rho_fit)

    coeffs = np.polyfit(x, y, 1)
    n_fit = coeffs[0]
    log10A_fit = coeffs[1]
    A_fit = 10.0**log10A_fit

    return n_fit, A_fit, z_fit, rho_fit


def make_plots(z, rho_SN, n_fit, A_fit, out_prefix):
    """
    ρ_SN とベストフィットのべき則 A (1+z)^n をプロットして保存する。
    """
    z = np.asarray(z)
    rho_SN = np.asarray(rho_SN)
    rho_fit_model = A_fit * (1.0 + z)**n_fit

    # 線形軸で rho_SN(z)
    plt.figure(figsize=(7, 5))
    plt.plot(z, rho_SN, label=r"$\rho_{\rm SN}(z)$")
    plt.plot(z, rho_fit_model, linestyle="--",
             label=rf"Fit: $A(1+z)^n$, $n={n_fit:.3f}$")
    plt.xlabel("z")
    plt.ylabel(r"$\rho_{\rm SN}(z)$ (arb. units)")
    plt.title("SN-mode energy density and power-law fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_SN_vs_z.png", dpi=150)
    plt.close()

    # log-log プロット
    mask_pos = rho_SN > 0.0
    z_pos = z[mask_pos]
    rho_pos = rho_SN[mask_pos]
    rho_fit_pos = rho_fit_model[mask_pos]

    plt.figure(figsize=(7, 5))
    plt.plot(np.log10(1.0 + z_pos),
             np.log10(rho_pos),
             ".", label=r"data: $\log \rho_{\rm SN}$")
    plt.plot(np.log10(1.0 + z_pos),
             np.log10(rho_fit_pos),
             "-", label="power-law fit")
    plt.xlabel(r"$\log_{10}(1+z)$")
    plt.ylabel(r"$\log_{10} \rho_{\rm SN}$")
    plt.title("Power-law behaviour of SN-mode energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_SN_loglog.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 178B: SN-mode power-law check for TFGR"
    )
    parser.add_argument("--z_max", type=float, default=2.0,
                        help="最大赤方偏移 z_max (default: 2.0)")
    parser.add_argument("--n_z", type=int, default=400,
                        help="z グリッドの分割数 (default: 400)")
    parser.add_argument("--z_min_fit", type=float, default=0.01,
                        help="べき則フィットに使う z の下限 (default: 0.01)")
    parser.add_argument("--z_max_fit", type=float, default=1.5,
                        help="べき則フィットに使う z の上限 (default: 1.5)")

    parser.add_argument("--w_star", type=float, default=-1.0,
                        help="SN モードの基底状態方程式 w_* (default: -1.0)")
    parser.add_argument("--s", type=float, default=0.5,
                        help="ソース項 S(z) ~ (1+z)^s の指数 s (default: 0.5)")
    parser.add_argument("--rho0", type=float, default=1.0,
                        help="z=0 における rho_SN(0) (正の任意定数, default: 1.0)")
    parser.add_argument("--S0", type=float, default=0.1,
                        help="ソース項の振幅 S0 (default: 0.1)")

    parser.add_argument("--out_prefix", type=str,
                        default="phase178_tfgr_SN_mode",
                        help="出力ファイルのプレフィックス (default: phase178_tfgr_SN_mode)")

    args = parser.parse_args()

    print("=== Phase 178B: SN-mode power-law check ===")
    print(f" z_max      = {args.z_max}")
    print(f" n_z        = {args.n_z}")
    print(f" z_fit      = [{args.z_min_fit}, {args.z_max_fit}]")
    print(f" w_star     = {args.w_star}")
    print(f" s (source) = {args.s}")
    print(f" rho0       = {args.rho0}")
    print(f" S0         = {args.S0}")
    print(f" out_prefix = {args.out_prefix}")
    print("==========================================")

    z = np.linspace(0.0, args.z_max, args.n_z)
    rho_SN = compute_rho_SN(z, w_star=args.w_star,
                            s=args.s, rho0=args.rho0, S0=args.S0)

    # べき則フィット
    n_fit, A_fit, z_fit, rho_fit = fit_power_law(
        z, rho_SN,
        z_min_fit=args.z_min_fit,
        z_max_fit=args.z_max_fit
    )

    # 結果の要約
    print("------ Power-law fit result ------")
    print(f" n_fit (rho_SN ~ (1+z)^n) = {n_fit:.4f}")
    print(f" A_fit                     = {A_fit:.4e}")
    # 参考として w_eff も表示（相互作用が無いと仮定した場合）
    w_eff = -1.0 + n_fit / 3.0
    print(f" w_eff (if treated as DE) = {w_eff:.4f}")
    print("----------------------------------")

    # CSV 出力
    df = pd.DataFrame({
        "z": z,
        "rho_SN": rho_SN,
        "rho_SN_norm": rho_SN / rho_SN[0] if rho_SN[0] != 0 else np.nan,
        "rho_SN_fit": A_fit * (1.0 + z)**n_fit,
        "log10_1pz": np.log10(1.0 + z),
        "log10_rho_SN": np.where(rho_SN > 0.0,
                                 np.log10(rho_SN),
                                 np.nan)
    })
    csv_path = f"{args.out_prefix}_rho_SN_profile.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Phase 178B] Saved CSV -> {csv_path}")

    # プロット
    make_plots(z, rho_SN, n_fit, A_fit, args.out_prefix)
    print(f"[Phase 178B] Saved plots -> {args.out_prefix}_rho_SN_vs_z.png, {args.out_prefix}_rho_SN_loglog.png")


if __name__ == "__main__":
    main()
