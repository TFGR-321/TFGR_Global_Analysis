#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 184: TFGR feedback H(z) / q(z) / j(z) diagnostics

・Phase118 以降の Φ_feedback(z) プロファイルを読み込み
・Φ_fb(z) から「時間場エネルギー密度 Ω_TF(z)」を構成
・フリードマン方程式
      H^2(z) = H0^2 [ Ω_r0 (1+z)^4 + Ω_m0 (1+z)^3 + Ω_TF(z) ]
  で H(z) を計算
・数値微分から減速度パラメータ q(z)，ジャーク j(z) を算出
・同じ (H0, Ω_m0, Ω_r0) の ΛCDM と比較プロット
・結果を CSV ＋ 図3枚に保存

Ω_TF(z) の具体的な形はまだ「仮設定」です：
    Ω_TF(z) = Ω_TF0 * [ Φ_fb(z) / Φ_fb(z=0) ]^eps_TF
eps_TF と Ω_TF0 は引数で調整可能です。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_feedback_csv(path):
    """
    TFGR フィードバック CSV を読み込み、
    z 列と Phi_feedback 系の列を自動検出して返す。
    """
    df = pd.read_csv(path)

    # z 列
    z_candidates = [c for c in df.columns if c.strip().lower() in ("z", "redshift")]
    if not z_candidates:
        raise RuntimeError("CSV 内に z / redshift 列が見つかりません。")
    z_col = z_candidates[0]

    # Phi_feedback 列候補
    phi_candidates = []
    for c in df.columns:
        lc = c.strip().lower()
        if "phi_fb" in lc or "phi_feedback" in lc or "phifeedback" in lc:
            phi_candidates.append(c)
    if not phi_candidates:
        raise RuntimeError("CSV 内に Phi_feedback 系の列が見つかりません。")

    phi_col = phi_candidates[0]

    z = df[z_col].to_numpy(dtype=float)
    phi = df[phi_col].to_numpy(dtype=float)

    # z 昇順にソート
    order = np.argsort(z)
    return z[order], phi[order], z_col, phi_col


def build_Omega_TF_from_phi(z_grid, z_phi, phi, Omega_TF0, eps_TF):
    """
    Φ_fb(z) から Ω_TF(z) を構成する簡易モデル。
    Ω_TF(z) = Ω_TF0 * ( |φ(z)| / |φ(0)| )^eps_TF
    """
    # 小さすぎる値への対策
    phi_abs = np.abs(phi)
    phi_abs[phi_abs == 0.0] = np.min(phi_abs[phi_abs > 0.0]) * 1e-6

    # z=0 近傍での φ を線形補間で取得
    z0 = 0.0
    if z0 < z_phi.min() or z0 > z_phi.max():
        # 範囲外なら、最小 z を "現在" とみなす
        z0 = z_phi.min()
    phi0 = np.interp(z0, z_phi, phi_abs)

    # 安全のため
    if phi0 <= 0.0:
        raise RuntimeError("Phi_feedback(z=0) が 0 以下になっています。正規化に失敗。")

    # φ(z) → Ω_TF(z)
    phi_interp = np.interp(z_grid, z_phi, phi_abs)
    ratio = phi_interp / phi0
    Omega_TF_z = Omega_TF0 * (ratio ** eps_TF)
    return Omega_TF_z


def H_from_components(z, H0, Omega_r0, Omega_m0, Omega_TF_z):
    """
    成分 Ω_r0, Ω_m0, Ω_TF(z) から H(z) を計算。
    """
    Ez2 = (
        Omega_r0 * (1.0 + z) ** 4
        + Omega_m0 * (1.0 + z) ** 3
        + Omega_TF_z
    )
    Ez2[Ez2 < 0] = np.nan  # もしもの時の保険
    return H0 * np.sqrt(Ez2)


def q_from_H(z, H):
    """
    H(z) から減速度パラメータ q(z) を計算。
      q(z) = -1 + (1+z)/H * dH/dz
    """
    dH_dz = np.gradient(H, z, edge_order=2)
    q = -1.0 + (1.0 + z) * dH_dz / H
    return q


def j_from_q(z, q):
    """
    q(z) からジャーク j(z) を計算。
      j(z) = q(z) * (2 q(z) + 1) + (1+z) dq/dz
    ΛCDM の場合は常に j=1 のはず。
    """
    dq_dz = np.gradient(q, z, edge_order=2)
    j = q * (2.0 * q + 1.0) + (1.0 + z) * dq_dz
    return j


def lcdm_components(z, H0, Omega_r0, Omega_m0):
    """
    同じ (H0, Ω_m0, Ω_r0) でフラット ΛCDM を構成。
    """
    Omega_L0 = 1.0 - Omega_m0 - Omega_r0
    Ez2 = (
        Omega_r0 * (1.0 + z) ** 4
        + Omega_m0 * (1.0 + z) ** 3
        + Omega_L0
    )
    H = H0 * np.sqrt(Ez2)
    q = -1.0 + (1.0 + z) * np.gradient(H, z, edge_order=2) / H
    j = np.ones_like(z)  # フラット ΛCDM では j ≡ 1
    return H, q, j, Omega_L0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 184: TFGR feedback-based H(z)/q(z)/j(z) diagnostics"
    )
    parser.add_argument("--tfgr_csv", required=True, help="TFGR feedback CSV (phase118 など)")
    parser.add_argument("--H0", type=float, default=70.0, help="H0 [km/s/Mpc]")
    parser.add_argument("--Omega_m0", type=float, default=0.3)
    parser.add_argument("--Omega_r0", type=float, default=1.0e-4)
    parser.add_argument(
        "--Omega_TF0",
        type=float,
        default=0.7,
        help="z=0 における TFGR 時間場成分の密度パラメータ (初期値)",
    )
    parser.add_argument(
        "--eps_TF",
        type=float,
        default=1.0,
        help="Ω_TF(z) ∝ [Φ_fb(z)/Φ_fb(0)]^eps_TF の指数",
    )
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=1.5)
    parser.add_argument("--n_z", type=int, default=400)
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="出力ファイル名プレフィックス (例: phase184_tfgr_feedback_diag)",
    )

    args = parser.parse_args()

    # 1. Φ_feedback プロファイルを読み込み
    z_phi, phi_fb, z_col, phi_col = load_feedback_csv(args.tfgr_csv)
    print("=== Phase 184: TFGR feedback diagnostics ===")
    print(f"TFGR CSV : {args.tfgr_csv}")
    print(f"z column : {z_col}")
    print(f"Phi col  : {phi_col}")
    print("-------------------------------------------")

    # 2. 計算用 z グリッド
    z = np.linspace(args.z_min, args.z_max, args.n_z)

    # 3. Φ_fb(z) → Ω_TF(z)
    Omega_TF_z = build_Omega_TF_from_phi(
        z_grid=z,
        z_phi=z_phi,
        phi=phi_fb,
        Omega_TF0=args.Omega_TF0,
        eps_TF=args.eps_TF,
    )

    # 4. TFGR H(z), q(z), j(z)
    H_tfgr = H_from_components(z, args.H0, args.Omega_r0, args.Omega_m0, Omega_TF_z)
    q_tfgr = q_from_H(z, H_tfgr)
    j_tfgr = j_from_q(z, q_tfgr)

    # 5. ΛCDM との比較
    H_lcdm, q_lcdm, j_lcdm, OmL0 = lcdm_components(
        z, args.H0, args.Omega_r0, args.Omega_m0
    )

    print(f"H0          = {args.H0:.3f} km/s/Mpc")
    print(f"Omega_m0    = {args.Omega_m0:.4f}")
    print(f"Omega_r0    = {args.Omega_r0:.4e}")
    print(f"Omega_TF0   = {args.Omega_TF0:.4f}")
    print(f"eps_TF      = {args.eps_TF:.3f}")
    print(f"LambdaCDM ΩΛ0 (for ref) = {OmL0:.4f}")
    print("-------------------------------------------")

    # 6. CSV 出力
    out_csv = f"{args.out_prefix}_Hqj_profile.csv"
    df_out = pd.DataFrame(
        {
            "z": z,
            "Omega_TF_z": Omega_TF_z,
            "H_tfgr": H_tfgr,
            "q_tfgr": q_tfgr,
            "j_tfgr": j_tfgr,
            "H_lcdm": H_lcdm,
            "q_lcdm": q_lcdm,
            "j_lcdm": j_lcdm,
        }
    )
    df_out.to_csv(out_csv, index=False)
    print(f"[INFO] Saved diagnostics CSV -> {out_csv}")

    # 7. 図の作成

    # (a) H(z)
    plt.figure(figsize=(7, 5))
    plt.plot(z, H_tfgr, label="TFGR H(z)")
    plt.plot(z, H_lcdm, "--", label="ΛCDM H(z)")
    plt.xlabel("z")
    plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("H(z) diagnostics: TFGR vs ΛCDM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig_H = f"{args.out_prefix}_H_z.png"
    plt.savefig(out_fig_H, dpi=150)
    plt.close()
    print(f"[INFO] Saved H(z) plot -> {out_fig_H}")

    # (b) q(z)
    plt.figure(figsize=(7, 5))
    plt.plot(z, q_tfgr, label="TFGR q(z)")
    plt.plot(z, q_lcdm, "--", label="ΛCDM q(z)")
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("z")
    plt.ylabel("q(z)")
    plt.title("Deceleration parameter q(z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig_q = f"{args.out_prefix}_q_z.png"
    plt.savefig(out_fig_q, dpi=150)
    plt.close()
    print(f"[INFO] Saved q(z) plot -> {out_fig_q}")

    # (c) j(z)
    plt.figure(figsize=(7, 5))
    plt.plot(z, j_tfgr, label="TFGR j(z)")
    plt.plot(z, j_lcdm, "--", label="ΛCDM j(z)=1")
    plt.xlabel("z")
    plt.ylabel("j(z)")
    plt.title("Jerk parameter j(z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig_j = f"{args.out_prefix}_j_z.png"
    plt.savefig(out_fig_j, dpi=150)
    plt.close()
    print(f"[INFO] Saved j(z) plot -> {out_fig_j}")

    print("=== Phase 184 完了: TFGR フィードバック H/q/j 診断 ===")


if __name__ == "__main__":
    main()
