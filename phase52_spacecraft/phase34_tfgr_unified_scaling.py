#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 34: TFGR unified scaling diagram

- 理論 TFGR モデル Δt(L) を log-log で描画
- これまで作成したミッション別 CSV（任意に存在するもの）を
  上に重ねてマーカー表示する

想定する既存 CSV:
  - nh_arrokoth_tfgr_fit.csv
  - vg1_jupiter_tfgr_fit_tfgr.csv
  - vg2_uranus_tfgr_fit_tfgr.csv
  （無くても OK。あれば自動で読み込んでプロットします）
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tfgr_model(L_m, dt0, Lc, p, q):
    """TFGR Δt(L) = dt0 * (1 + (L/Lc)^p)^q"""
    L_m = np.asarray(L_m, dtype=float)
    return dt0 * (1.0 + (L_m / Lc) ** p) ** q


def detect_column(df, candidates):
    """候補名リストから、最初に見つかった列名を返す（無ければ None）"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_dataset(ax, csv_path, label, color, marker):
    """
    既存 CSV から代表点（中央値）を 1 点プロット。
    ファイルが無ければ何もしない。
    """
    if not os.path.exists(csv_path):
        print(f"⚠ {csv_path} が見つからないためスキップします。")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠ {csv_path} の読み込みに失敗: {e}")
        return

    # 距離列の推定
    L_col = detect_column(df, ["L_m", "L", "L_km"])
    if L_col is None:
        print(f"⚠ {csv_path}: 距離列(L_m, L, L_km) が見つからずスキップします。")
        return

    L = df[L_col].astype(float)

    # L_km の場合は m に変換
    if L_col == "L_km":
        L_m = L * 1.0e3
    else:
        L_m = L

    # Δt 列の推定
    dt_col = detect_column(
        df,
        ["dt_res", "dt_tfgr", "Delta_t", "delta_t", "tfgr_dt"],
    )
    if dt_col is None:
        print(f"⚠ {csv_path}: Δt 列が見つからずスキップします。")
        return

    dt = df[dt_col].astype(float)

    # NaN 除去
    mask = np.isfinite(L_m) & np.isfinite(dt)
    L_m = L_m[mask]
    dt = dt[mask]

    if len(L_m) == 0:
        print(f"⚠ {csv_path}: 有効なデータ点が無くスキップします。")
        return

    # 代表値として中央値を使用
    L_med = np.median(L_m)
    dt_med = np.median(dt)

    ax.scatter(L_med, dt_med, color=color, marker=marker, s=80, label=label)
    # 注釈（少し右上にオフセット）
    ax.annotate(
        label,
        xy=(L_med, dt_med),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

    print(f"✅ {csv_path}: L_med = {L_med:.3e} m, dt_med = {dt_med:.3e} s をプロット")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 34: TFGR unified scaling diagram"
    )
    parser.add_argument(
        "--Lc", type=float, default=4.0e9,
        help="Critical length scale Lc [m] (default: 4.0e9)"
    )
    parser.add_argument(
        "--p", type=float, default=0.21,
        help="TFGR exponent p (default: 0.21)"
    )
    parser.add_argument(
        "--q", type=float, default=1.32,
        help="TFGR exponent q (default: 1.32)"
    )
    parser.add_argument(
        "--dt0", type=float, default=5.184e-19,
        help="Baseline Δt0 [s] (default: 5.184e-19 from optical link)"
    )
    parser.add_argument(
        "--out", type=str, default="tfgr_unified_scaling",
        help="Output prefix (default: tfgr_unified_scaling)"
    )
    args = parser.parse_args()

    Lc = args.Lc
    p = args.p
    q = args.q
    dt0 = args.dt0

    print("=== TFGR Unified Scaling Diagram ===")
    print(f"Lc  = {Lc:.3e} m")
    print(f"p   = {p}")
    print(f"q   = {q}")
    print(f"dt0 = {dt0:.3e} s")

    # 1) 理論カーブ：10^-6 m 〜 10^13 m まで対数スイープ
    L_min = 1.0e-6      # 1 micron
    L_max = 1.0e13      # 宇宙論スケールの手前
    L_grid = np.logspace(np.log10(L_min), np.log10(L_max), 2000)
    dt_grid = tfgr_model(L_grid, dt0, Lc, p, q)

    # 2) 図の設定
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(
        L_grid, dt_grid,
        color="black", lw=2.0,
        label="TFGR model"
    )

    # 臨界長 Lc の縦線
    ax.axvline(Lc, color="gray", ls="--", lw=1)
    ax.text(
        Lc, dt_grid[0] * 2,
        "L_c", rotation=90,
        va="bottom", ha="right",
        fontsize=9, color="gray"
    )

    # 3) 既知実験のポイントを追加 --------------------

    # (a) 光学リンク（2220 km, 7e-17）
    L_opt_m = 2220.0e3
    dt_opt = 5.184e-19  # 既に使っている値
    ax.scatter(L_opt_m, dt_opt, color="tab:green", marker="s", s=70, label="Optical link (2220 km)")
    ax.annotate(
        "Optical link",
        xy=(L_opt_m, dt_opt),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

    # (b) Rosetta RPC-ICA（代表値：L ~ 5e8 m, dt ~ 4e-13 s）
    #     厳密な CSV をまだ直接結合していないので、ここは理論＋フィット値から代表点を置く。
    L_ros_m = 5.0e8
    dt_ros = 4.0e-13
    ax.scatter(L_ros_m, dt_ros, color="tab:purple", marker="D", s=70, label="Rosetta RPC-ICA (repr.)")
    ax.annotate(
        "Rosetta RPC-ICA",
        xy=(L_ros_m, dt_ros),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

    # (c) New Horizons – Arrokoth
    add_dataset(
        ax,
        "nh_arrokoth_tfgr_fit.csv",
        "New Horizons – Arrokoth",
        color="tab:blue",
        marker="o",
    )

    # (d) Voyager 1 – Jupiter
    add_dataset(
        ax,
        "vg1_jupiter_tfgr_fit_tfgr.csv",
        "Voyager 1 – Jupiter",
        color="tab:red",
        marker="^",
    )

    # (e) Voyager 2 – Uranus
    add_dataset(
        ax,
        "vg2_uranus_tfgr_fit_tfgr.csv",
        "Voyager 2 – Uranus",
        color="tab:orange",
        marker="v",
    )

    # 軸ラベルなど
    ax.set_xlabel("Distance L [m]", fontsize=12)
    ax.set_ylabel("Δt [s]", fontsize=12)
    ax.set_title("TFGR Unified Scaling Diagram", fontsize=15)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_png = f"{args.out}.png"
    out_csv = f"{args.out}_curve.csv"

    fig.savefig(out_png, dpi=200)
    print(f"✅ 図を保存しました: {out_png}")

    # 理論カーブも CSV で保存
    df_curve = pd.DataFrame({"L_m": L_grid, "dt_tfgr": dt_grid})
    df_curve.to_csv(out_csv, index=False)
    print(f"✅ 理論カーブを保存しました: {out_csv}")


if __name__ == "__main__":
    main()
