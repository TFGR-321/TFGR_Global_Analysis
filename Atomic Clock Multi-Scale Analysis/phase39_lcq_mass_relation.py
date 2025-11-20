#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase39_lcq_mass_relation.py

TFGR Phase 39:
  L_cQ vs m_eff (log–log) 関係の可視化と線形回帰解析。

入力:
  - waic_mass_scaling.txt（Phase 38 出力）

出力:
  - lcq_mass_relation.png : log–log プロット＋回帰線
  - lcq_mass_fit_summary.txt : フィット係数と相関情報
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse
import os


def parse_mass_scaling_txt(path):
    """
    Phase 38 の出力テキストから m_eff [u] と L_cQ [m] を抽出。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    m_vals = []
    lc_vals = []

    lc_match = re.search(r"LcQ0.*?=\s*([\d\.Ee\+\-]+)", "".join(lines))
    lcq0_base = float(lc_match.group(1)) if lc_match else np.nan

    in_eff = False
    for line in lines:
        if "Effective masses" in line:
            in_eff = True
            continue
        if in_eff:
            m = re.search(r"([\w\-\+_]+):.*?([\d\.]+)\s*u", line)
            if m:
                pair_name = m.group(1)
                m_eff = float(m.group(2))
                # LcQ = LcQ0 * (100 / m_eff)
                lcq = lcq0_base * (100.0 / m_eff)
                m_vals.append(m_eff)
                lc_vals.append(lcq)

    return np.array(m_vals), np.array(lc_vals)


def main():
    parser = argparse.ArgumentParser(description="Phase 39: log–log relation LcQ vs m_eff.")
    parser.add_argument("--txt", required=True, help="waic_mass_scaling.txt")
    parser.add_argument("--out", default="output_phase39_lcq_mass", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    m_vals, lc_vals = parse_mass_scaling_txt(args.txt)
    if len(m_vals) == 0:
        raise RuntimeError("m_eff データが抽出できませんでした。")

    log_m = np.log10(m_vals)
    log_lc = np.log10(lc_vals)

    slope, intercept, r, p, stderr = linregress(log_m, log_lc)

    # 回帰式
    xfit = np.linspace(min(log_m)-0.1, max(log_m)+0.1, 100)
    yfit = intercept + slope * xfit

    # 可視化
    plt.figure(figsize=(6,5))
    plt.scatter(log_m, log_lc, color="royalblue", label="TFGR (Phase 38) data")
    plt.plot(xfit, yfit, color="darkorange", label=f"fit: logLc = {intercept:.2f} + {slope:.2f} log m")
    plt.xlabel("log₁₀(m_eff [u])")
    plt.ylabel("log₁₀(L_cQ [m])")
    plt.title("TFGR Quantum Critical Scale vs Atomic Mass")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lcq_mass_relation.png"), dpi=300)

    # テキスト出力
    out_txt = os.path.join(args.out, "lcq_mass_fit_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Phase 39: TFGR LcQ–mass relation (log–log fit)\n\n")
        f.write(f"Fit equation: log10(L_cQ) = {intercept:.4f} + {slope:.4f} * log10(m_eff)\n")
        f.write(f"Correlation r = {r:.3f},  p = {p:.3e}\n")
        f.write(f"Slope stderr = {stderr:.4f}\n\n")
        f.write(f"Derived power law:  L_cQ ∝ m_eff^{slope:.2f}\n")

    print(f"✅ 結果を {out_txt} と lcq_mass_relation.png に出力しました。")
    print(f"推定スロープ b = {slope:.2f} （理論値 −1 と比較）")


if __name__ == "__main__":
    main()
