#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 52E:
Dual-layer time-field curvature (Quantum layer + Planetary layer)
-----------------------------------------------------------------
量子スケール(光格子時計)と惑星スケール(GPS+LLR)を
別々の L 領域に分けて TFGR フィットし、それぞれの臨界長 Lc と
時間場曲率ゼロ点(∂²Φ_t/∂L²=0)を推定する。

モデル:
    Δt(L) = A * ( [1 + (L/Lc)^p]^q - 1 )

出力:
  - {out}_dual_layerfit.png
  - {out}_dual_layer_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -------------------------------------------------
# 基本ユーティリティ
# -------------------------------------------------
def safe_load_L_dt(path, L_col=None, dt_col=None):
    """
    CSV から (L, dt) を読み込む。
    - 数値列のみ使用
    - NaN / inf / L<=0 を除外
    - L の重複を除外
    """
    df = pd.read_csv(path)

    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.empty:
        raise ValueError(f"{path}: 数値列が見つかりません。")

    # L 列選択
    if L_col is not None and L_col in num_df.columns:
        L = num_df[L_col].to_numpy()
    else:
        L = num_df.iloc[:, 0].to_numpy()

    # Δt 列選択
    if dt_col is not None and dt_col in num_df.columns:
        dt = num_df[dt_col].to_numpy()
    else:
        if num_df.shape[1] >= 2:
            dt = num_df.iloc[:, 1].to_numpy()
        else:
            dt = num_df.iloc[:, 0].to_numpy()

    L = np.asarray(L, dtype=float)
    dt = np.asarray(dt, dtype=float)

    mask = np.isfinite(L) & np.isfinite(dt) & (L > 0.0)
    L, dt = L[mask], dt[mask]

    if L.size == 0:
        raise ValueError(f"{path}: 有効な (L, dt) データがありません。")

    # 重複 L を除外
    uniq_L, uniq_idx = np.unique(L, return_index=True)
    L = uniq_L
    dt = dt[uniq_idx]

    order = np.argsort(L)
    return L[order], dt[order]


def concat_group(paths, L_col=None, dt_col=None):
    """複数CSVを読み込んで単純結合し，再ソート・重複除外した (L, dt) を返す"""
    L_all, dt_all = [], []
    for p in paths:
        if not p.strip():
            continue
        L, dt = safe_load_L_dt(p, L_col=L_col, dt_col=dt_col)
        L_all.append(L)
        dt_all.append(dt)

    if not L_all:
        raise ValueError("concat_group: 有効なCSVパスがありません。")

    L_cat = np.concatenate(L_all)
    dt_cat = np.concatenate(dt_all)

    order = np.argsort(L_cat)
    L_cat, dt_cat = L_cat[order], dt_cat[order]

    uniq_L, uniq_idx = np.unique(L_cat, return_index=True)
    L_cat = uniq_L
    dt_cat = dt_cat[uniq_idx]

    return L_cat, dt_cat


def tfgr_dt(L, A, Lc, p, q):
    """TFGR モデル Δt(L)"""
    L = np.asarray(L, dtype=float)
    return A * ((1.0 + (L / Lc) ** p) ** q - 1.0)


def compute_derivatives(L, y):
    """一次・二次微分"""
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.gradient(y, L)
    d2y = np.gradient(dy, L)
    return dy, d2y


def detect_zero_crossing(L, f):
    """f(L) の符号が反転する最初の位置を線形補間で返す"""
    L = np.asarray(L, dtype=float)
    f = np.asarray(f, dtype=float)
    sign = np.sign(f)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, x1 = L[i], L[i + 1]
    y0, y1 = f[i], f[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def fit_tfgr(L, dt, p_fix, q_fix):
    """
    Δt(L) に TFGR をフィットし，(A_hat, Lc_hat) を返す。
    Lc > 0 となるよう bounds を設定。
    """
    L = np.asarray(L, dtype=float)
    dt = np.asarray(dt, dtype=float)

    mask = np.isfinite(L) & np.isfinite(dt)
    L_fit, dt_fit = L[mask], dt[mask]

    if L_fit.size < 3:
        raise ValueError("fit_tfgr: フィットに十分な点数がありません。")

    L_min, L_max = float(L_fit.min()), float(L_fit.max())
    Lc0 = np.sqrt(L_min * L_max)

    denom = (1.0 + (L_max / Lc0) ** p_fix) ** q_fix - 1.0
    if denom == 0.0:
        denom = 1e-12
    A0 = (dt_fit.max() - dt_fit.min()) / denom

    def model(Lx, A, Lc):
        return tfgr_dt(Lx, A, Lc, p_fix, q_fix)

    popt, _ = curve_fit(
        model,
        L_fit,
        dt_fit,
        p0=[A0, Lc0],
        bounds=([-np.inf, 1e-6], [np.inf, 1e12]),
        maxfev=20000,
    )
    return popt  # A_hat, Lc_hat


# -------------------------------------------------
# メイン
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 52E: Dual-layer TFGR time-field curvature"
    )
    parser.add_argument("--quantum_csv", required=True,
                        help="Comma-separated quantum-scale CSVs (e.g. phase50E_du.csv)")
    parser.add_argument("--gps_csv", required=True,
                        help="GPS CSV (e.g. AJAC_phase51B_tfgr_input.csv)")
    parser.add_argument("--llr_csv", required=True,
                        help="Comma-separated LLR CSVs")
    parser.add_argument("--L_col", default=None,
                        help="Column name for L (optional)")
    parser.add_argument("--dt_col", default=None,
                        help="Column name for Δt (optional)")
    parser.add_argument("--p", type=float, default=0.21,
                        help="TFGR exponent p (default 0.21)")
    parser.add_argument("--q", type=float, default=1.32,
                        help="TFGR exponent q (default 1.32)")
    parser.add_argument("--scan_points", type=int, default=1500,
                        help="Number of log-spaced L points (default 1500)")
    parser.add_argument("--quantum_Lmax", type=float, default=1e3,
                        help="Upper L [m] for quantum layer fit (default 1e3)")
    parser.add_argument("--planetary_Lmin", type=float, default=1e5,
                        help="Lower L [m] for planetary layer fit (default 1e5)")
    parser.add_argument("--out", required=True,
                        help="Output prefix (e.g. phase52E_AJAC)")
    args = parser.parse_args()

    p_fix, q_fix = args.p, args.q

    # ---- データ読み込み ----
    quantum_paths = [s.strip() for s in args.quantum_csv.split(",") if s.strip()]
    L_q_all, dt_q_all = concat_group(quantum_paths, L_col=args.L_col, dt_col=args.dt_col)

    gps_paths = [args.gps_csv.strip()]
    llr_paths = [s.strip() for s in args.llr_csv.split(",") if s.strip()]
    L_p_all, dt_p_all = concat_group(gps_paths + llr_paths,
                                     L_col=args.L_col, dt_col=args.dt_col)

    # ---- 領域分離 ----
    q_mask = L_q_all <= args.quantum_Lmax
    if q_mask.sum() < 3:
        # データが足りなければ全域を使う
        L_q_layer, dt_q_layer = L_q_all, dt_q_all
    else:
        L_q_layer, dt_q_layer = L_q_all[q_mask], dt_q_all[q_mask]

    p_mask = L_p_all >= args.planetary_Lmin
    if p_mask.sum() < 3:
        L_p_layer, dt_p_layer = L_p_all, dt_p_all
    else:
        L_p_layer, dt_p_layer = L_p_all[p_mask], dt_p_all[p_mask]

    # ---- 各層 TFGR フィット ----
    A_q, Lc_q = fit_tfgr(L_q_layer, dt_q_layer, p_fix, q_fix)
    A_p, Lc_p = fit_tfgr(L_p_layer, dt_p_layer, p_fix, q_fix)

    # ---- スキャン軸 ----
    L_all = np.concatenate([L_q_all, L_p_all])
    L_min, L_max = float(L_all.min()), float(L_all.max())
    L_scan = np.logspace(np.log10(L_min / 10.0), np.log10(L_max * 10.0),
                         args.scan_points)

    # モデル Δt
    dt_q_model = tfgr_dt(L_scan, A_q, Lc_q, p_fix, q_fix)
    dt_p_model = tfgr_dt(L_scan, A_p, Lc_p, p_fix, q_fix)

    # 正規化（最大絶対値でスケール合わせ）
    norm_q = np.max(np.abs(dt_q_model)) or 1.0
    norm_p = np.max(np.abs(dt_p_model)) or 1.0
    dt_q_norm = dt_q_model / norm_q
    dt_p_norm = dt_p_model / norm_p

    # 勾配・曲率
    dphi_q, d2phi_q = compute_derivatives(L_scan, dt_q_model)
    dphi_p, d2phi_p = compute_derivatives(L_scan, dt_p_model)

    # 曲率ゼロ点
    Lz_q = detect_zero_crossing(L_scan, d2phi_q)
    Lz_p = detect_zero_crossing(L_scan, d2phi_p)

    # ---- プロット ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax0, ax1, ax2 = axes

    # 正規化 Δt(L)
    ax0.plot(L_scan, dt_q_norm, label="Quantum layer (norm.)", color="blue")
    ax0.plot(L_scan, dt_p_norm, label="Planetary layer (norm.)", color="orange", linestyle="--")
    ax0.set_xscale("log")
    ax0.set_ylabel(r"Normalized $\Delta t(L)$")
    ax0.set_title("Dual-layer TFGR fit & time-field curvature")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # 一次微分
    ax1.plot(L_scan, dphi_q, color="blue", label="Quantum")
    ax1.plot(L_scan, dphi_p, color="orange", linestyle="--", label="Planetary")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\partial \Phi_t / \partial L$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 二次微分
    ax2.plot(L_scan, d2phi_q, color="blue", label="Quantum")
    ax2.plot(L_scan, d2phi_p, color="orange", linestyle="--", label="Planetary")
    ax2.axvline(Lc_q, color="blue", linestyle=":", label=f"Lc_q ≈ {Lc_q:.2e} m")
    ax2.axvline(Lc_p, color="orange", linestyle=":", label=f"Lc_p ≈ {Lc_p:.2e} m")
    if Lz_q is not None:
        ax2.axvline(Lz_q, color="blue", linestyle="--", label=f"L_zero_q ≈ {Lz_q:.2e} m")
    if Lz_p is not None:
        ax2.axvline(Lz_p, color="orange", linestyle="--", label=f"L_zero_p ≈ {Lz_p:.2e} m")
    ax2.set_xscale("log")
    ax2.set_xlabel("L [m]")
    ax2.set_ylabel(r"$\partial^2 \Phi_t / \partial L^2$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{args.out}_dual_layerfit.png", dpi=200)

    # ---- サマリ CSV ----
    rows = [
        {
            "group": "quantum_layer",
            "A_hat": A_q,
            "Lc_hat_m": Lc_q,
            "L_zero_curvature_m": Lz_q,
            "L_min_layer": float(L_q_layer.min()),
            "L_max_layer": float(L_q_layer.max()),
        },
        {
            "group": "planetary_layer",
            "A_hat": A_p,
            "Lc_hat_m": Lc_p,
            "L_zero_curvature_m": Lz_p,
            "L_min_layer": float(L_p_layer.min()),
            "L_max_layer": float(L_p_layer.max()),
        },
    ]
    pd.DataFrame(rows).to_csv(f"{args.out}_dual_layer_summary.csv", index=False)

    print("✅ Phase 52E dual-layer fit finished.")
    print(f"   Quantum layer:   Lc ≈ {Lc_q:.3e} m, L_zero ≈ {Lz_q if Lz_q is not None else 'N/A'}")
    print(f"   Planetary layer: Lc ≈ {Lc_p:.3e} m, L_zero ≈ {Lz_p if Lz_p is not None else 'N/A'}")


if __name__ == "__main__":
    main()
