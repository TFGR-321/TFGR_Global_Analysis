#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 42 : Time-field curvature tensor ∇²Φ_t(L, m)
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt

def load_params(summary_path):
    p = {}
    text = open(summary_path, "r", encoding="utf-8").read()
    def grab(name):
        m = re.search(rf"{name}\s*=\s*([\-0-9.eE\+]+)", text)
        return float(m.group(1)) if m else np.nan
    for k in ["dt0_M","Lc_M","p_M","q_M","dt0_Q","LcQ0","p_Q","q_Q","L_int_ref"]:
        p[k] = grab(k)
    return p

def delta_t_tfgr(L, L_int_ref, m_eff,
                 dt0_M, Lc_M, p_M, q_M,
                 dt0_Q, LcQ0, p_Q, q_Q, m0=100.0):
    term_M = dt0_M * (1 + (L/Lc_M)**p_M)**q_M
    Lc_Q_m = LcQ0 * (m0/m_eff)
    term_Q = dt0_Q * (1 + (L_int_ref/Lc_Q_m)**p_Q)**q_Q
    return term_M + term_Q

def main():
    summary_path = "output_phase40_phi_t/phi_t_mass_summary.txt"
    out_dir = "output_phase42_phi_t_curvature"
    os.makedirs(out_dir, exist_ok=True)
    p = load_params(summary_path)

    # Grid
    L_vals = np.logspace(-3, 8, 300)
    m_vals = np.linspace(80, 200, 150)
    Phi = np.zeros((len(m_vals), len(L_vals)))
    for i, m in enumerate(m_vals):
        for j, L in enumerate(L_vals):
            Phi[i,j] = delta_t_tfgr(L, p["L_int_ref"], m,
                                    p["dt0_M"], p["Lc_M"], p["p_M"], p["q_M"],
                                    p["dt0_Q"], p["LcQ0"], p["p_Q"], p["q_Q"])

    # Normalize
    Phi_norm = Phi/np.max(np.abs(Phi))
    logL = np.log10(L_vals)
    dL = np.gradient(logL)
    dm = np.gradient(m_vals)

    # 2nd derivatives
    d2Phi_dL2 = np.gradient(np.gradient(Phi_norm, axis=1), axis=1) / (dL**2)
    d2Phi_dm2 = np.gradient(np.gradient(Phi_norm, axis=0), axis=0) / (dm[:,None]**2)
    laplacian = d2Phi_dL2 + d2Phi_dm2  # ∇²Φ_t

    X, Y = np.meshgrid(logL, m_vals)

    # --- Figure 1: heatmap ---
    plt.figure(figsize=(7,5))
    im = plt.pcolormesh(X, Y, laplacian, cmap="coolwarm", shading="auto")
    plt.colorbar(im, label="∇²Φ_t (normalized)")
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Time-field curvature map ∇²Φ_t(L, m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"phi_t_curvature_map.png"), dpi=300)
    plt.close()

    # --- Figure 2: contour ---
    plt.figure(figsize=(7,5))
    cs = plt.contour(X, Y, laplacian, levels=np.linspace(-0.5,0.5,21), cmap="coolwarm")
    plt.clabel(cs, inline=1, fontsize=8)
    plt.xlabel("log10(L [m])")
    plt.ylabel("m_eff [u]")
    plt.title("Contours of ∇²Φ_t(L, m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"phi_t_curvature_contour.png"), dpi=300)
    plt.close()

    # --- Summary ---
    with open(os.path.join(out_dir,"phi_t_curvature_summary.txt"),"w",encoding="utf-8") as f:
        f.write("# Phase 42: curvature tensor of Φ_t(L, m)\n\n")
        f.write(f"max(∇²Φ_t) = {np.max(laplacian):.3e}\n")
        f.write(f"min(∇²Φ_t) = {np.min(laplacian):.3e}\n")
        idx_max = np.unravel_index(np.argmax(laplacian), laplacian.shape)
        idx_min = np.unravel_index(np.argmin(laplacian), laplacian.shape)
        f.write(f"max at: L={L_vals[idx_max[1]]:.3e} m, m={m_vals[idx_max[0]]:.1f} u\n")
        f.write(f"min at: L={L_vals[idx_min[1]]:.3e} m, m={m_vals[idx_min[0]]:.1f} u\n")

    print("✅ 出力完了:", out_dir)

if __name__ == "__main__":
    main()
