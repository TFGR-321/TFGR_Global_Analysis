#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def guess_columns(df, L_col=None, dt_col=None):
    """
    入力CSVから L列・Δt列を推定するヘルパー。
    明示的に --L_col / --dt_col が指定されていればそれを優先。
    """

    # 明示指定があればそれを使う
    if L_col is not None and L_col in df.columns:
        L_name = L_col
    else:
        # L列の候補をざっくり推定
        L_name = None
        for c in df.columns:
            cl = c.lower()
            if any(key in cl for key in ["l_m", "distance", "radius", "r_m"]):
                L_name = c
                break
        if L_name is None:
            # 数値列の先頭をLとして使う最後の手段
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    L_name = c
                    break

    if dt_col is not None and dt_col in df.columns:
        dt_name = dt_col
    else:
        # Δt列の候補を推定
        dt_candidates = []
        for c in df.columns:
            cl = c.lower()
            if any(key in cl for key in ["dt", "delta_t", "t_res", "clock"]):
                dt_candidates.append(c)
        dt_name = dt_candidates[0] if dt_candidates else None

        if dt_name is None:
            # 数値列の2番目をΔtとして使う最後の手段
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                dt_name = num_cols[1]
            elif len(num_cols) == 1:
                dt_name = num_cols[0]

    if L_name is None or dt_name is None:
        raise ValueError(
            f"Could not infer L or dt columns. "
            f"L_col guessed: {L_name}, dt_col guessed: {dt_name}. "
            f"Please specify them explicitly with --L_col and --dt_col."
        )

    return L_name, dt_name


def compute_phi_and_derivatives(L, dt, c, dt0=None):
    """
    L (距離) と Δt (時間残差) から
      Φ_t = c^2 * Δt / Δt0
    を構成し、L に関する一次・二次微分を計算する。

    dt0 は正規化定数（デフォルト：|Δt|の中央値）。
    これはスケーリングだけなので、極小位置や符号反転位置には影響しない。
    """

    L = np.asarray(L, dtype=float)
    dt = np.asarray(dt, dtype=float)

    # Lでソートし、重複Lは1つに縮約
    order = np.argsort(L)
    L_sorted = L[order]
    dt_sorted = dt[order]

    unique_L, unique_idx = np.unique(L_sorted, return_index=True)
    L_unique = L_sorted[unique_idx]
    dt_unique = dt_sorted[unique_idx]

    if dt0 is None:
        # 規格化：|dt|の中央値を1オーダーの値に
        dt0 = np.median(np.abs(dt_unique))
        if dt0 == 0:
            dt0 = 1.0

    phi = (c ** 2) * dt_unique / dt0

    # L による一次・二次微分（非一様間隔も numpy.gradient が処理）
    dphi_dL = np.gradient(phi, unique_L)
    d2phi_dL2 = np.gradient(dphi_dL, unique_L)

    return unique_L, phi, dphi_dL, d2phi_dL2


def summarize_curvature(L, dphi_dL, d2phi_dL2):
    """
    曲率構造の簡単な要約量を返す:
      - L_min, L_max
      - |dPhi/dL| 最小点とその値
      - |d²Phi/dL²| 最小点とその値（= 曲率 ~ 0 に最も近い点）
    """

    L = np.asarray(L, dtype=float)
    dphi_dL = np.asarray(dphi_dL, dtype=float)
    d2phi_dL2 = np.asarray(d2phi_dL2, dtype=float)

    if len(L) == 0:
        return {
            "L_min": np.nan,
            "L_max": np.nan,
            "L_at_min_abs_dphi": np.nan,
            "min_abs_dphi": np.nan,
            "L_at_min_abs_d2phi": np.nan,
            "min_abs_d2phi": np.nan,
        }

    L_min = float(L.min())
    L_max = float(L.max())

    idx_min_abs_dphi = int(np.argmin(np.abs(dphi_dL)))
    idx_min_abs_d2phi = int(np.argmin(np.abs(d2phi_dL2)))

    return {
        "L_min": L_min,
        "L_max": L_max,
        "L_at_min_abs_dphi": float(L[idx_min_abs_dphi]),
        "min_abs_dphi": float(dphi_dL[idx_min_abs_dphi]),
        "L_at_min_abs_d2phi": float(L[idx_min_abs_d2phi]),
        "min_abs_d2phi": float(d2phi_dL2[idx_min_abs_d2phi]),
    }


def make_curvature_plot(datasets, out_prefix):
    """
    datasets: dict[name] = {"L":..., "dphi":..., "d2phi":...}
    各データセットについて dPhi/dL, d²Phi/dL² を L に対して描画。
    """

    if not datasets:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1, ax2 = axes

    for name, data in datasets.items():
        L = data["L"]
        dphi = data["dphi"]
        d2phi = data["d2phi"]

        ax1.plot(L, dphi, label=name)
        ax2.plot(L, d2phi, label=name)

    ax1.set_ylabel(r"$\partial \Phi_t / \partial L$")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r"$L$ [m]")
    ax2.set_ylabel(r"$\partial^2 \Phi_t / \partial L^2$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Time-field gradient and curvature vs. L")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"{out_prefix}_curvature_vs_L.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_gradient_signmap(datasets, out_prefix):
    """
    dPhi/dL の符号を L–dataset の「符号マップ」として可視化する:
      - 青: 負
      - 白: ~0
      - 赤: 正
    """

    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(datasets)))

    sc = None
    for j, (name, data) in enumerate(datasets.items()):
        L = np.asarray(data["L"], dtype=float)
        dphi = np.asarray(data["dphi"], dtype=float)
        if len(L) == 0:
            continue

        signs = np.sign(dphi)
        # -1 -> 0, 0 -> 0.5, +1 -> 1 にマッピング
        vals = np.where(signs > 0, 1.0, np.where(signs < 0, 0.0, 0.5))

        sc = ax.scatter(L, np.full_like(L, j), c=vals, cmap="bwr",
                        vmin=0, vmax=1, s=10)

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(list(datasets.keys()))
    ax.set_xlabel(r"$L$ [m]")
    ax.set_title("Sign of $\\partial \\Phi_t / \partial L$")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, label="sign(dPhi_t/dL)")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["negative", "zero", "positive"])

    fig.tight_layout()
    out_path = f"{out_prefix}_gradient_signmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 52: Time-field curvature analysis. "
            "Compute Phi_t(L), its first and second derivatives, "
            "and extract critical-scale summaries."
        )
    )
    parser.add_argument(
        "--gps_csv",
        type=str,
        default=None,
        help="GPS CSV file (e.g., AJAC_phase51B_tfgr_input.csv)",
    )
    parser.add_argument(
        "--llr_csv",
        type=str,
        default=None,
        help=(
            "Comma-separated list of LLR CSV files "
            "(e.g., apollo11_*.csv,apollo14_*.csv,apollo15_*.csv)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help=(
            "Output prefix. For example, '--out phase52_AJAC' will produce "
            "phase52_AJAC_curvature_vs_L.png, "
            "phase52_AJAC_gradient_signmap.png, "
            "phase52_AJAC_summary_curvature.csv."
        ),
    )
    parser.add_argument(
        "--L_col",
        type=str,
        default=None,
        help="(Optional) Column name for L in the input CSVs. If omitted, guessed.",
    )
    parser.add_argument(
        "--dt_col",
        type=str,
        default=None,
        help="(Optional) Column name for Δt in the input CSVs. If omitted, guessed.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=299792458.0,
        help="Speed of light [m/s]. Default: 299792458.0",
    )

    args = parser.parse_args()

    datasets = {}
    summary_rows = []

    def process_file(path):
        path = path.strip()
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path)
        L_name, dt_name = guess_columns(df, L_col=args.L_col, dt_col=args.dt_col)

        df_clean = df[[L_name, dt_name]].dropna()
        if df_clean.empty:
            raise ValueError(f"No valid (L, dt) data in file: {path}")

        L = df_clean[L_name].values
        dt = df_clean[dt_name].values

        L_u, phi, dphi, d2phi = compute_phi_and_derivatives(
            L, dt, c=args.c, dt0=None
        )

        dataset_name = Path(path).stem
        datasets[dataset_name] = {
            "L": L_u,
            "phi": phi,
            "dphi": dphi,
            "d2phi": d2phi,
        }

        summary = summarize_curvature(L_u, dphi, d2phi)
        summary["dataset"] = dataset_name
        summary_rows.append(summary)

    # GPS
    if args.gps_csv is not None:
        process_file(args.gps_csv)

    # LLR (カンマ区切り)
    if args.llr_csv is not None:
        for item in args.llr_csv.split(","):
            if item.strip():
                process_file(item)

    if not datasets:
        raise RuntimeError("No input data provided. Use --gps_csv and/or --llr_csv.")

    # プロット生成
    make_curvature_plot(datasets, args.out)
    make_gradient_signmap(datasets, args.out)

    # サマリーCSV
    summary_df = pd.DataFrame(summary_rows)
    cols = [
        "dataset",
        "L_min",
        "L_max",
        "L_at_min_abs_dphi",
        "min_abs_dphi",
        "L_at_min_abs_d2phi",
        "min_abs_d2phi",
    ]
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    summary_path = f"{args.out}_summary_curvature.csv"
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
