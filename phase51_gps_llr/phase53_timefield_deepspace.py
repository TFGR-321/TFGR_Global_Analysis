#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 53: Deep-space TFGR time-field curvature
----------------------------------------------

複数スケール (惑星〜惑星間〜カイパーベルト) の Δt(L) データを結合し，
時間場 Φ_t(L) の勾配・曲率を L に対して解析する。

想定データ:
  - inner_csv:  惑星/地球近傍スケール (例: tfgr_unified_scaling_curve.csv)
  - deep_csv:   深宇宙スケール (例: nh_arrokoth_tfgr_fit.csv, vg*_tfgr_fit_tfgr.csv, rpcica_tfgr_ready_v2.csv)

各 CSV には少なくとも
  * L ~ 距離 [m] (列名は L, dist, distance, r, radius, au など)
  * Δt または TFGR 残差 (列名は dt, delta_t, res, tfgr など)
の 2 つの数値列が含まれていることを想定する。

出力:
  - {out}_deepspace_curvature.png
      上:   Δt(L) vs L (log-log)
      中:   ∂Φ_t/∂L vs L
      下:   ∂²Φ_t/∂L² vs L と曲率ゼロ点
  - {out}_deepspace_summary.csv
      global および各データセットの L 範囲・曲率ゼロ点などの数値表
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------
# データ読込ユーティリティ
# -------------------------------------------------

def guess_L_dt_columns(df, L_col_hint=None, dt_col_hint=None):
    """
    DataFrame から L と Δt の列名を推定する。
    ヒントが指定されていればそれを優先。
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("数値列が存在しません。")

    # ヒント優先
    if L_col_hint in df.columns:
        L_col = L_col_hint
    else:
        # L 候補
        lname = [c.lower() for c in df.columns]
        L_candidates = []
        key_L = ["l", "dist", "distance", "radius", "r", "au"]
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in key_L):
                L_candidates.append(c)
        L_col = L_candidates[0] if L_candidates else num_cols[0]

    if dt_col_hint in df.columns:
        dt_col = dt_col_hint
    else:
        # Δt 候補
        key_dt = ["dt", "delta", "res", "tfgr", "phi"]
        dt_candidates = []
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in key_dt):
                dt_candidates.append(c)
        # L と同じ列は除外
        dt_candidates = [c for c in dt_candidates if c != L_col]
        if dt_candidates:
            dt_col = dt_candidates[0]
        else:
            # 数値列から L_col を除いて最初の列
            others = [c for c in num_cols if c != L_col]
            dt_col = others[0] if others else num_cols[0]

    return L_col, dt_col


def load_L_dt_from_csv(path, L_col_hint=None, dt_col_hint=None):
    """
    CSV ファイルから (L, dt) を読み込む。
    - 数値列のみ使用
    - NaN, inf, L<=0 を除外
    - L 重複行は 1 つにまとめる（先頭値を採用）
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: CSV が空です。")

    L_col, dt_col = guess_L_dt_columns(df, L_col_hint, dt_col_hint)

    L = df[L_col].to_numpy(dtype=float)
    dt = df[dt_col].to_numpy(dtype=float)

    mask = np.isfinite(L) & np.isfinite(dt) & (L > 0.0)
    L = L[mask]
    dt = dt[mask]

    if L.size == 0:
        raise ValueError(f"{path}: 有効な L>0, dt データがありません。")

    # L でソート & 重複排除
    order = np.argsort(L)
    L = L[order]
    dt = dt[order]

    uniq_L, idx = np.unique(L, return_index=True)
    L = uniq_L
    dt = dt[idx]

    return L, dt, L_col, dt_col


# -------------------------------------------------
# 数値微分 & 曲率ゼロ点検出
# -------------------------------------------------

def compute_derivatives(L, y):
    """
    不等間隔格子 L 上の y(L) に対して一次・二次微分を計算。
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    dy_dL = np.gradient(y, L)
    d2y_dL2 = np.gradient(dy_dL, L)
    return dy_dL, d2y_dL2


def detect_zero_crossing(L, f):
    """
    f(L) の符号が変わる点を検出し，
    最初の交差点を線形補間で求めて返す。
    見つからない場合は None。
    """
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


# -------------------------------------------------
# メイン処理
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 53: Deep-space TFGR time-field curvature"
    )
    parser.add_argument(
        "--inner_csv",
        default="",
        help="内側スケール CSV（カンマ区切りで複数指定可，例: tfgr_unified_scaling_curve.csv）",
    )
    parser.add_argument(
        "--deep_csv",
        required=True,
        help="深宇宙スケール CSV（カンマ区切り，例: nh_arrokoth_tfgr_fit.csv,vg1_jupiter_tfgr_fit_tfgr.csv,vg2_uranus_tfgr_fit_tfgr.csv）",
    )
    parser.add_argument(
        "--L_col",
        default=None,
        help="L 列名を固定したい場合に指定（すべてのファイルで共通なら有効）",
    )
    parser.add_argument(
        "--dt_col",
        default=None,
        help="Δt 列名を固定したい場合に指定（すべてのファイルで共通なら有効）",
    )
    parser.add_argument(
        "--scan_points",
        type=int,
        default=0,
        help="0 の場合は生の L グリッドで解析、それ以外なら logspace に再サンプル（任意）",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="出力ファイルのプレフィックス（例: phase53_deepspace）",
    )
    args = parser.parse_args()

    # --------- 入力ファイルリスト構築 --------- #
    datasets = []

    if args.inner_csv.strip():
        inner_paths = [s.strip() for s in args.inner_csv.split(",") if s.strip()]
        for p in inner_paths:
            datasets.append(("inner:" + Path(p).stem, p))

    deep_paths = [s.strip() for s in args.deep_csv.split(",") if s.strip()]
    for p in deep_paths:
        datasets.append(("deep:" + Path(p).stem, p))

    if not datasets:
        raise RuntimeError("解析対象 CSV が指定されていません。")

    # --------- 各 CSV を読み込んで結合 --------- #
    all_L = []
    all_dt = []
    summary_rows = []

    print(">>> Loading datasets...")
    for label, path in datasets:
        L, dt, L_name, dt_name = load_L_dt_from_csv(
            path, L_col_hint=args.L_col, dt_col_hint=args.dt_col
        )
        all_L.append(L)
        all_dt.append(dt)
        summary_rows.append(
            {
                "dataset": label,
                "file": path,
                "L_col_used": L_name,
                "dt_col_used": dt_name,
                "L_min": float(L.min()),
                "L_max": float(L.max()),
                "N_points": int(L.size),
            }
        )
        print(
            f"  - {label}: {path}  "
            f"[L: {L_name}, dt: {dt_name}, N={L.size}, "
            f"L_min={L.min():.3e}, L_max={L.max():.3e}]"
        )

    # 結合
    L_all = np.concatenate(all_L)
    dt_all = np.concatenate(all_dt)

    # ソート & 重複削除
    order = np.argsort(L_all)
    L_all = L_all[order]
    dt_all = dt_all[order]

    uniq_L, idx = np.unique(L_all, return_index=True)
    L_all = uniq_L
    dt_all = dt_all[idx]

    # --------- 必要なら共通 log グリッドに再サンプル --------- #
    if args.scan_points and args.scan_points > 10:
        L_min, L_max = float(L_all.min()), float(L_all.max())
        L_grid = np.logspace(np.log10(L_min), np.log10(L_max), args.scan_points)
        # 補間（単純線形，外側は最近傍外挿）
        dt_grid = np.interp(L_grid, L_all, dt_all)
        L_use, dt_use = L_grid, dt_grid
    else:
        L_use, dt_use = L_all, dt_all

    # --------- 勾配・曲率計算 --------- #
    dphi_dL, d2phi_dL2 = compute_derivatives(L_use, dt_use)

    # 曲率ゼロ点検出
    L_zero = detect_zero_crossing(L_use, d2phi_dL2)
    idx_min_abs = int(np.argmin(np.abs(d2phi_dL2)))
    L_min_abs = float(L_use[idx_min_abs])
    min_abs_d2 = float(d2phi_dL2[idx_min_abs])

    # グローバル行を summary に追加
    summary_rows.append(
        {
            "dataset": "GLOBAL",
            "file": "(all)",
            "L_col_used": "(mixed)",
            "dt_col_used": "(mixed)",
            "L_min": float(L_use.min()),
            "L_max": float(L_use.max()),
            "N_points": int(L_use.size),
            "L_zero_curvature_estimate": L_zero,
            "L_at_min_abs_curvature": L_min_abs,
            "min_abs_d2Phi_dL2": min_abs_d2,
        }
    )

    # --------- プロット --------- #
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax0, ax1, ax2 = axes

    # 上: Δt(L)
    ax0.loglog(L_use, np.abs(dt_use))
    ax0.set_ylabel(r"$|\Delta t(L)|$")
    ax0.set_title("Deep-space time-field curvature (Phase 53)")
    ax0.grid(True, which="both", alpha=0.3)

    # 中: ∂Φ_t/∂L （Δt の形を見るだけなのでスケールは気にしない）
    ax1.plot(L_use, dphi_dL)
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\partial \Phi_t / \partial L \propto \partial \Delta t/\partial L$")
    ax1.grid(True, which="both", alpha=0.3)

    # 下: ∂²Φ_t/∂L² とゼロ点
    ax2.plot(L_use, d2phi_dL2)
    ax2.set_xscale("log")
    ax2.set_xlabel("L [m]")
    ax2.set_ylabel(r"$\partial^2 \Phi_t / \partial L^2$")
    ax2.grid(True, which="both", alpha=0.3)

    if L_zero is not None:
        ax2.axvline(L_zero, color="red", linestyle="--",
                    label=f"curvature zero ≈ {L_zero:.2e} m")
    ax2.axvline(L_min_abs, color="orange", linestyle=":",
                label=f"min |curvature| ≈ {L_min_abs:.2e} m")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{args.out}_deepspace_curvature.png", dpi=200)

    # --------- サマリ保存 --------- #
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.out}_deepspace_summary.csv", index=False)

    print("\n>>> Done.")
    if L_zero is not None:
        print(f"  Curvature zero (sign change)  L ≈ {L_zero:.3e} m")
    print(
        f"  Min |curvature|              L ≈ {L_min_abs:.3e} m, "
        f"|d2Phi/dL2| ≈ {min_abs_d2:.3e}"
    )
    print(f"  Figure : {args.out}_deepspace_curvature.png")
    print(f"  Summary: {args.out}_deepspace_summary.csv")


if __name__ == "__main__":
    main()
