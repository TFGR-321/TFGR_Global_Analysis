#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 138: TFGR Temporal Mode Stability Analysis
-------------------------------------------------
固有モードの安定性解析、および共鳴遷移点 k_res の検出を行う。

入力：
  --matrix_csv : モード相互作用行列 I(k_i, k_j)

出力：
  ・固有値スペクトル (eigenvalues)
  ・支配モード k_dom
  ・有効モード数 N_eff
  ・累積パワー 50% 点
  ・summary CSV
  ・図（固有値スペクトル、固有ベクトルパワー）
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. データ読み込み
# -------------------------------------------------------

def load_matrix(csv_path):
    df = pd.read_csv(csv_path)
    mat = df.values
    return mat

# -------------------------------------------------------
# 2. 固有値解析
# -------------------------------------------------------

def eigen_analysis(mat):
    eigvals, eigvecs = np.linalg.eig(mat)

    # 固有値を実部でソート
    idx = np.argsort(-eigvals.real)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs

# -------------------------------------------------------
# 3. モードパワー解析
# -------------------------------------------------------

def mode_power(eigvec):
    """ 固有ベクトルのパワー P(k) を返す """
    return np.abs(eigvec)**2 / np.sum(np.abs(eigvec)**2)

def analyse_spectrum(eigvals, eigvecs):
    # 支配モード
    dom_vec = eigvecs[:, 0]
    P = mode_power(dom_vec)

    k_dom = np.argmax(P)
    frac_dom = P[k_dom]

    # 有効モード数
    N_eff = 1.0 / np.sum(P**2)

    # 50% 累積パワー点
    cum = np.cumsum(P)
    k_50 = np.where(cum >= 0.5)[0][0]

    return k_dom, frac_dom, N_eff, k_50, P

# -------------------------------------------------------
# 4. メイン処理
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_csv", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    print("=== Phase 138: TFGR Temporal Mode Stability Analysis ===")
    print(f"Loading matrix from: {args.matrix_csv}")

    mat = load_matrix(args.matrix_csv)
    eigvals, eigvecs = eigen_analysis(mat)

    print(f"[INFO] Eigenvalues computed: N={len(eigvals)}")

    k_dom, frac_dom, N_eff, k_50, P = analyse_spectrum(eigvals, eigvecs)

    print("---------------------------------------")
    print(f"Dominant mode k_dom     = {k_dom}")
    print(f"Fraction in k_dom       = {frac_dom:.4f}")
    print(f"Effective # of modes    = {N_eff:.3f}")
    print(f"k at 50% cumulative     = {k_50}")
    print("---------------------------------------")

    # Save results
    summary = pd.DataFrame({
        "k_dom":[k_dom],
        "frac_dom":[frac_dom],
        "N_eff":[N_eff],
        "k_50":[k_50]
    })
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    # 固有値図
    plt.figure(figsize=(6,4))
    plt.plot(eigvals.real, eigvals.imag, "o")
    plt.xlabel("Re eigenvalue")
    plt.ylabel("Im eigenvalue")
    plt.title("Eigenvalue Spectrum")
    plt.grid()
    plt.savefig(f"{args.out_prefix}_eigvals.png", dpi=200)

    # パワースペクトル
    plt.figure(figsize=(7,4))
    plt.stem(P)
    plt.title("Power Spectrum of Dominant Eigenmode")
    plt.xlabel("Mode index k")
    plt.ylabel("Power")
    plt.grid()
    plt.savefig(f"{args.out_prefix}_power.png", dpi=200)

    print(f"[INFO] Saved outputs with prefix '{args.out_prefix}'")


if __name__ == "__main__":
    main()
