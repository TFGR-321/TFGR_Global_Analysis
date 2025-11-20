#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase41_phi_t_gradient_map.py

Phase 41:
  Mass-dependent Time-Field Potential Φ_t(L, m)
  → 勾配 ∂Φ_t/∂L, ∂Φ_t/∂m の解析と可視化

入力:
  - output_phase40_phi_t/phi_t_mass_summary.txt
  - 同ディレクトリ内の Phi_t 計算関数（同一フォルダで実行可）

出力:
  - phi_t_grad_map.png
  - phi_t_grad_magnitude.png
  - phi_t_grad_summary.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re

def load_phase40_params(summary_path):
    params = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()
    def grab(name):
        m = re.search(rf"{name}\s*=\s*([\-0-9.eE\+]+)", text)
        return float(m.group(1)) if m else np.nan
    params["dt0_M"] = grab("dt0_M")
    params["Lc_M"]  = grab("Lc_M")
    params["p_M"]   = grab("p_M")
    params["q_M"]   = grab("q_M")
    params["dt0_Q"] = grab("dt0_Q")
    params["LcQ0"]  = grab("LcQ0")
    params["p_Q"]   = grab("p_Q")
    params["q_Q"]   = grab("q_Q")
    params["L_int_ref"] = grab("L_int_ref")
    return params

def delta_t_tfgr(L, L_int_ref, m_eff,
                 dt0_M, Lc_M, p_M, q_M,
                 dt0_Q, LcQ0, p_Q, q_Q, m0=100.0):
    term_M = dt0_M * (1.0 + (L / Lc_M)**p_M)**q_M
    Lc_Q_m = LcQ0 * (m0 / m_eff)
    term_Q = dt0_Q * (1.0 + (L_int_ref / Lc_Q_m)**p_Q)**q_Q
    return term_M + term_Q

def main():
    summary_path = "output_phase40_phi_t/phi_t_mass_summary.txt"
    out_dir = "output_phase41_phi_t_grad"
    os.makedirs(out_dir, exist_ok=True)
    p = load_phase40_params(summary_path)

    # L, m グリッド
    L_vals = np.logspace(-3, 8, 200)
    m_vals = np.linspace(80, 200, 120)
    Phi = np.zeros((len(m_vals), len(L_vals)))

    for i, m in enumerate(m_vals):
        for j, L in enumerate(L_vals):
            Phi[i, j] = delta_t_tfgr(L, p["L_int_ref"], m,
                                     p["dt0_M"], p["Lc_M"], p["p_M"], p["q_M"],
                                     p["dt0_Q"], p["LcQ0"], p["p_Q"], p["q_Q"])

    Phi_norm = Phi / np.max(np.abs(Phi))

    # 勾配計算
    dL = np.gradient(np.log10(L_vals))
    dm = np.gradient(m_vals)
    dPhi_dL = np.gradient(Phi_norm, axis=1) / dL
    dPhi_dm = np.gradient(Phi_norm, axis=0) / dm[:, None]
    grad_mag = np.sqrt(dPhi_dL**2 + dPhi_dm**2)

    X, Y = np.meshgrid(np.log10(L_vals), m_vals)

    # ベクトル場（サンプリング密度下げて描画）
    skip = (slice(None, None, 8), slice(None, None, 8))
    plt.figure(figsize=(7,5))
    plt.quiver(X[skip], Y[skip], dPhi_dL[skip], dPhi_dm[skip],
               grad_mag[skip], cmap="plasma", scale=20)
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Gradient field of Φ_t(L, m)")
    plt.colorbar(label="|∇Φ_t|")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phi_t_grad_map.png"), dpi=300)
    plt.close()

    # 勾配強度ヒートマップ
    plt.figure(figsize=(7,5))
    im = plt.pcolormesh(X, Y, grad_mag, shading="auto", cmap="viridis")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Gradient magnitude |∇Φ_t(L, m)|")
    plt.colorbar(im, label="|∇Φ_t| (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phi_t_grad_magnitude.png"), dpi=300)
    plt.close()

    # summary 出力
    with open(os.path.join(out_dir, "phi_t_grad_summary.txt"), "w", encoding="utf-8") as f:
        f.write("# Phase 41: Gradient analysis of Φ_t(L, m)\n\n")
        f.write(f"Max |∇Φ_t| = {np.max(grad_mag):.3e}\n")
        f.write(f"Min |∇Φ_t| = {np.min(grad_mag):.3e}\n")
        idx_max = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
        f.write(f"At max gradient: L = {L_vals[idx_max[1]]:.3e} m, m_eff = {m_vals[idx_max[0]]:.1f} u\n")

    print(f"✅ 出力完了: {out_dir}/phi_t_grad_map.png, phi_t_grad_magnitude.png, phi_t_grad_summary.txt")

if __name__ == "__main__":
    main()
