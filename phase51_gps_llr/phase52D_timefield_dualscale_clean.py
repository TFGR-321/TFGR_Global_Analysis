#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 52D CLEAN:
Dual-scale TFGR time-field curvature (quantum + planetary, interpolation-free)
-----------------------------------------------------------------------------

量子スケール (phase50E_du.csv など) と惑星スケール (GPS + LLR) の
Δt(L) を TFGR 型

    Δt(L) = A * ( [1 + (L/Lc)^p]^q - 1 )

にフィット。Φ_t(L) の一次・二次微分を計算し、
量子側 Lc_q と惑星側 Lc_p、および曲率ゼロ点 L_zero_q, L_zero_p を求める。

出力:
  - {out}_dualscale_curvature.png
  - {out}_dualscale_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------- 基本ユーティリティ ---------- #

def safe_load_L_dt(path, L_col=None, dt_col=None):
    """
    CSV から (L, dt) を読み込む。
    - 数値列だけ使う
    - NaN / inf / L<=0 を除去
    - L の重複を削除
    """
    df = pd.read_csv(path)

    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.empty:
        raise ValueError(f"{path}: 数値列が見つかりません。")

    # L 列
    if L_col is not None and L_col in num_df.columns:
        L = num_df[L_col].to_numpy()
    else:
        L = num_df.iloc[:, 0].to_numpy()

    # Δt 列
    if dt_col is not None and dt_col in num_df.columns:
        dt = num_df[dt_col].to_numpy()
    else:
        if num_df.shape[1] >= 2:
            dt = num_df.iloc[:, 1].to_numpy()
        else:
            dt = num_df.iloc[:, 0].to_numpy()

    L = np.asarray(L, dtype=float)
    dt = np.asarray(dt, dtype=float)

    mask = np.isfinite(L) & np.isfinite(dt) & (L > 0)
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
    """
    パスのリストから (L, dt) を読み込み、単純結合。
    """
    L_all = []
    dt_all = []
    for p in paths:
        if not p.strip():
            continue
        L, dt = safe_load_L_dt(p, L_col=L_col, dt_col=dt_col)
        L_all.append(L)
        dt_all.append(dt)
    if not L_all:
        raise ValueError("concat_group: 有効ファイルがありません。")

    L_cat = np.concatenate(L_all)
    dt_cat = np.concatenate(dt_all)

    # 再度ソート & 重複除去
    order = np.argsort(L_cat)
    L_cat = L_cat[order]
    dt_cat = dt_cat[order]

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
    Δt(L) に TFGR モデルをフィット。
    Lc > 0 となるよう bounds をかける。
    """
    L = np.asarray(L, dtype=float)
    dt = np.asarray(dt, dtype=float)

    mask = np.isfinite(L) & np.isfinite(dt)
    L_fit = L[mask]
    dt_fit = dt[mask]

    if L_fit.size < 3:
        raise ValueError("fit_tfgr: データ点が少なすぎます。")

    L_min, L_max = float(L_fit.min()), float(L_fit.max())
    Lc0 = np.sqrt(L_min * L_max)

    denom = (1.0 + (L_max / Lc0) ** p_fix) ** q_fix - 1.0
    if denom == 0:
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
    return popt  # (A_hat, Lc_hat)


# ---------- メイン ---------- #

def main():
    parser = argparse.ArgumentParser(
        description="Phase 52D CLEAN: Dual-scale TFGR time-field curvature"
    )
    parser.add_argument("--quantum_csv", required=True,
                        help="Comma-separated quantum-scale CSVs (e.g. phase50E_du.csv)")
    parser.add_argument("--gps_csv", required=True,
                        help="GPS CSV file (e.g. AJAC_phase51B_tfgr_input.csv)")
    parser.add_argument("--llr_csv", required=True,
                        help="Comma-separated LLR CSV files")
    parser.add_argument("--L_col", default=None,
                        help="Name of L column (optional)")
    parser.add_argument("--dt_col", default=None,
                        help="Name of Δt column (optional)")
    parser.add_argument("--p", type=float, default=0.21,
                        help="TFGR exponent p (default 0.21)")
    parser.add_argument("--q", type=float, default=1.32,
                        help="TFGR exponent q (default 1.32)")
    parser.add_argument("--scan_points", type=int, default=1500,
                        help="Number of log-spaced L scan points")
    parser.add_argument("--out", required=True,
                        help="Output prefix, e.g. phase52D_AJAC")
    args = parser.parse_args()

    p_fix, q_fix = args.p, args.q

    # --- 量子スケール --- #
    quantum_paths = [s.strip() for s in args.quantum_csv.split(",") if s.strip()]
    L_q, dt_q = concat_group(quantum_paths, L_col=args.L_col, dt_col=args.dt_col)

    # --- 惑星スケール (GPS + LLR) --- #
    gps_paths = [args.gps_csv.strip()]
    llr_paths = [s.strip() for s in args.llr_csv.split(",") if s.strip()]
    planetary_paths = gps_paths + llr_paths
    L_p, dt_p = concat_group(planetary_paths, L_col=args.L_col, dt_col=args.dt_col)

    # --- TFGR フィット --- #
    A_q, Lc_q = fit_tfgr(L_q, dt_q, p_fix, q_fix)
    A_p, Lc_p = fit_tfgr(L_p, dt_p, p_fix, q_fix)

    # --- スキャン軸 --- #
    L_all = np.concatenate([L_q, L_p])
    L_min, L_max = float(L_all.min()), float(L_all.max())
    L_scan = np.logspace(np.log10(L_min / 10.0), np.log10(L_max * 10.0), args.scan_points)

    # モデル Δt
    dt_q_model = tfgr_dt(L_scan, A_q, Lc_q, p_fix, q_fix)
    dt_p_model = tfgr_dt(L_scan, A_p, Lc_p, p_fix, q_fix)

    # 勾配・曲率
    dphi_q, d2phi_q = compute_derivatives(L_scan, dt_q_model)
    dphi_p, d2phi_p = compute_derivatives(L_scan, dt_p_model)

    # 曲率ゼロ点
    Lz_q = detect_zero_crossing(L_scan, d2phi_q)
    Lz_p = detect_zero_crossing(L_scan, d2phi_p)

    # --- プロット --- #
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax0, ax1, ax2 = axes

    # Δt(L)
    ax0.plot(L_scan, dt_q_model, label="Quantum TFGR", color="blue")
    ax0.plot(L_scan, dt_p_model, label="Planetary TFGR", color="orange", linestyle="--")
    ax0.set_xscale("log")
    ax0.set_ylabel("Δt(L)")
    ax0.set_title("Dual-scale TFGR Fit & Time-field Curvature (CLEAN)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # ∂Φt/∂L
    ax1.plot(L_scan, dphi_q, color="blue")
    ax1.plot(L_scan, dphi_p, color="orange", linestyle="--")
    ax1.set_xscale("log")
    ax1.set_ylabel("∂Φₜ/∂L")
    ax1.grid(True, alpha=0.3)

    # ∂²Φt/∂L²
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
    ax2.set_ylabel("∂²Φₜ/∂L²")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{args.out}_dualscale_curvature.png", dpi=200)

    # --- サマリ CSV --- #
    rows = [
        {
            "group": "quantum",
            "A_hat": A_q,
            "Lc_hat_m": Lc_q,
            "L_zero_curvature_m": Lz_q,
            "L_min_data": float(L_q.min()),
            "L_max_data": float(L_q.max()),
        },
        {
            "group": "planetary",
            "A_hat": A_p,
            "Lc_hat_m": Lc_p,
            "L_zero_curvature_m": Lz_p,
            "L_min_data": float(L_p.min()),
            "L_max_data": float(L_p.max()),
        },
    ]
    pd.DataFrame(rows).to_csv(f"{args.out}_dualscale_summary.csv", index=False)

    print("✅ Dual-scale TFGR curvature finished.")
    print(f"   quantum:  Lc ≈ {Lc_q:.3e} m, L_zero ≈ {Lz_q if Lz_q is not None else 'N/A'}")
    print(f"   planetary: Lc ≈ {Lc_p:.3e} m, L_zero ≈ {Lz_p if Lz_p is not None else 'N/A'}")


if __name__ == "__main__":
    main()
