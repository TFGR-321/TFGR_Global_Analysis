# phase31_tfgr_newhorizons_geometry.py
#
# New Horizons SPICE カーネルから
#  - New Horizons (ID: -98)
#  - ターゲット (初期値 486958 Arrokoth)
# 間の距離 L(t) を計算して CSV と図に出力するスクリプトです。
#
# 使い方（例）:
#   python phase31_tfgr_newhorizons_geometry.py ^
#       --kernel_dir spice_nh ^
#       --start 2019-01-01T04:00:00 ^
#       --stop  2019-01-01T08:00:00 ^
#       --step 60 ^
#       --out nh_arrokoth_test
#
# 必要ライブラリ:
#   pip install spiceypy numpy pandas matplotlib

import os
import glob
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spiceypy as spice


def load_kernels(kernel_dir: str) -> List[str]:
    """
    指定ディレクトリ内の SPICE カーネルを一括ロードする。

    対象: *.bsp, *.bc, *.bds, *.tpc*, *.tsc*, *.ti*, *.tf*, *.sclk*, *.txt
    （New Horizons の配布セットをだいたいカバー）
    """
    patterns = [
        "*.bsp", "*.bc", "*.bds",
        "*.tpc", "*.tpc.txt",
        "*.tsc", "*.tsc.txt",
        "*.ti", "*.ti.txt",
        "*.tf", "*.tf.txt",
        "*.sclk", "*.sclk.tsc",
        "*.txt",   # 念のため
    ]

    loaded = []
    for pat in patterns:
        for path in glob.glob(os.path.join(kernel_dir, pat)):
            # 同じファイルを二重にロードしない
            path = os.path.abspath(path)
            if path in loaded:
                continue
            try:
                spice.furnsh(path)
                loaded.append(path)
            except Exception as e:
                print(f"⚠ カーネル読み込み失敗: {path} -> {e}")

    print(f"✅ ロードしたカーネル数: {len(loaded)}")
    for p in loaded:
        print("   -", os.path.basename(p))
    return loaded


def compute_geometry(
    target: str,
    observer: str,
    frame: str,
    abcorr: str,
    start_utc: str,
    stop_utc: str,
    step_sec: float,
) -> pd.DataFrame:
    """
    SPICE を用いて target–observer 間の距離 L(t) を計算する。

    Parameters
    ----------
    target : str
        NAIF ID もしくはボディ名（例 "486958", "JUPITER" など）
    observer : str
        観測者（New Horizons は NAIF ID -98）
    frame : str
        参照フレーム（通常 "J2000"）
    abcorr : str
        光行差補正指定（例 "LT+S"）
    start_utc, stop_utc : str
        期間の UTC 文字列（"YYYY-MM-DDTHH:MM:SS"）
    step_sec : float
        サンプリング間隔 [秒]
    """
    et_start = spice.str2et(start_utc)
    et_stop = spice.str2et(stop_utc)

    if et_stop <= et_start:
        raise ValueError("stop 時刻は start より後にしてください。")

    n_step = int(np.floor((et_stop - et_start) / step_sec)) + 1
    ets = et_start + np.arange(n_step) * step_sec

    times_utc = []
    distances_km = []

    print(f"=== 幾何計算 ===")
    print(f"ターゲット : {target}")
    print(f"観測者     : {observer}")
    print(f"期間       : {start_utc} ～ {stop_utc}")
    print(f"ステップ   : {step_sec} s, 点数 {n_step}")

    for et in ets:
        try:
            pos, lt = spice.spkpos(target, et, frame, abcorr, observer)
        except Exception as e:
            # カバレッジ外などの場合はスキップ
            print(f"⚠ et={et} で spkpos 失敗: {e}")
            continue

        r_km = np.linalg.norm(pos)
        utc = spice.et2utc(et, "ISOC", 3)
        times_utc.append(utc)
        distances_km.append(r_km)

    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(times_utc),
            "et": ets[: len(times_utc)],
            "L_km": distances_km,
        }
    )
    df["L_m"] = df["L_km"] * 1000.0
    df["L_AU"] = df["L_km"] / 1.495978707e8  # 1 AU ≒ 1.495978707e8 km

    return df


def plot_distance(df: pd.DataFrame, out_prefix: str) -> None:
    """距離 vs 時刻のプロットを保存する。"""
    if df.empty:
        print("⚠ データが空なのでプロットをスキップします。")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_utc"], df["L_km"], marker="o", ms=3)
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Distance L [km]")
    ax.set_title("New Horizons – Target Distance")
    fig.autofmt_xdate()
    ax.grid(True, ls="--", alpha=0.4)

    png_path = f"{out_prefix}_distance.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"✅ 図を保存しました: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="New Horizons SPICE カーネルから幾何（距離 L(t)）を計算するスクリプト"
    )
    parser.add_argument(
        "--kernel_dir",
        default=".",
        help="SPICE カーネルが置いてあるディレクトリ（既定: カレント）",
    )
    parser.add_argument(
        "--target",
        default="486958",  # Arrokoth の NAIF ID
        help='ターゲット天体（NAIF ID か名前, 既定: "486958" = Arrokoth）',
    )
    parser.add_argument(
        "--observer",
        default="-98",  # New Horizons
        help='観測者（既定: "-98" = New Horizons）',
    )
    parser.add_argument(
        "--frame",
        default="J2000",
        help='参照フレーム（既定: "J2000"）',
    )
    parser.add_argument(
        "--abcorr",
        default="LT+S",
        help='光行差補正指定（既定: "LT+S"）',
    )
    parser.add_argument(
        "--start",
        default="2019-01-01T04:00:00",
        help='開始時刻 UTC (例: "2019-01-01T04:00:00")',
    )
    parser.add_argument(
        "--stop",
        default="2019-01-01T08:00:00",
        help='終了時刻 UTC (例: "2019-01-01T08:00:00")',
    )
    parser.add_argument(
        "--step",
        type=float,
        default=60.0,
        help="サンプリング間隔 [秒]（既定: 60）",
    )
    parser.add_argument(
        "--out",
        default="newhorizons_tfgr_arrokoth",
        help="出力ファイルのプレフィックス（既定: newhorizons_tfgr_arrokoth）",
    )

    args = parser.parse_args()

    # 1) カーネルをロード
    print("=== SPICE カーネルのロード ===")
    kernel_dir = os.path.abspath(args.kernel_dir)
    if not os.path.isdir(kernel_dir):
        raise FileNotFoundError(f"kernel_dir がディレクトリではありません: {kernel_dir}")
    loaded = load_kernels(kernel_dir)

    if not loaded:
        print("⚠ カーネルが 1 つも読み込めていません。kernel_dir を確認してください。")

    try:
        # 2) 幾何計算
        df = compute_geometry(
            target=args.target,
            observer=args.observer,
            frame=args.frame,
            abcorr=args.abcorr,
            start_utc=args.start,
            stop_utc=args.stop,
            step_sec=args.step,
        )

        print("\n=== 計算結果プレビュー ===")
        print(df.head())
        print("\n=== 距離統計量 [km] ===")
        print(df["L_km"].describe())

        # 3) CSV 保存
        csv_path = f"{args.out}_distance.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 幾何データを保存しました: {csv_path}")

        # 4) 図を保存
        plot_distance(df, args.out)

    finally:
        # カーネルをアンロード
        spice.kclear()
        print("SPICE カーネルをアンロードしました。")


if __name__ == "__main__":
    main()
