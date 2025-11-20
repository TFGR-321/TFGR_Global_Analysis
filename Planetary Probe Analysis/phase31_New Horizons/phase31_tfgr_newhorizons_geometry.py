# phase31_tfgr_newhorizons_geometry.py
#
# New Horizons SPICE カーネルから Target との距離 L(t) を計算する完全版スクリプト
# SPICE カーネルのロード順は NASA/JPL 推奨順に従い、
# Leap Second → Frames → PCK → SCLK → SPK → CK → DSK
# の順で確実にロードされるように修正。
#
# 実行例:
#   python phase31_tfgr_newhorizons_geometry.py \
#       --kernel_dir spice_nh \
#       --start 2019-01-01T04:00:00 \
#       --stop  2019-01-01T08:00:00 \
#       --step 60 \
#       --target "2014 MU69" \
#       --out nh_arrokoth_test

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spiceypy as spice


# ================================================================
#  NASA/JPL 推奨順で SPICE カーネルを確実にロードする関数
# ================================================================
def load_kernels(kernel_dir: str):
    """
    SPICE カーネルを NASA/JPL 推奨順で確実にロードする。
    """
    print("=== Loading SPICE kernels ===")

    # NASA official recommended order
    exts_order = [
        (".tls",  "LSK  (Leap Seconds)"),
        (".tf",   "FK   (Frames)"),
        (".tpc",  "PCK  (Planet constants)"),
        (".tsc",  "SCLK (Spacecraft clock)"),
        (".bsp",  "SPK  (Ephemeris)"),
        (".bc",   "CK   (Orientation)"),
        (".bds",  "DSK  (Shape/Other binary)"),
    ]

    loaded = []

    for ext, label in exts_order:
        for fname in os.listdir(kernel_dir):
            if fname.lower().endswith(ext):
                path = os.path.join(kernel_dir, fname)
                try:
                    spice.furnsh(path)
                    loaded.append(path)
                    print(f"Loaded [{label}]: {fname}")
                except Exception as e:
                    print(f"⚠ Error loading {fname}: {e}")

    print(f"\nTotal loaded kernels: {len(loaded)}")
    return loaded


# ================================================================
# 幾何計算
# ================================================================
def compute_geometry(target, observer, frame, abcorr, start_utc, stop_utc, step_sec):

    et_start = spice.str2et(start_utc)
    et_stop = spice.str2et(stop_utc)

    if et_stop <= et_start:
        raise ValueError("stop 時刻は start より後にしてください。")

    n_step = int(np.floor((et_stop - et_start) / step_sec)) + 1
    ets = et_start + np.arange(n_step) * step_sec

    print(f"\n=== 幾何計算 ===")
    print(f"ターゲット : {target}")
    print(f"観測者     : {observer}")
    print(f"期間       : {start_utc} ～ {stop_utc}")
    print(f"ステップ   : {step_sec} s, 点数 {n_step}")

    times_utc = []
    distances_km = []
    ets_valid = []

    for et in ets:
        try:
            # pos[km], lt[s]
            pos, lt = spice.spkpos(target, et, frame, abcorr, observer)
        except Exception as e:
            print(f"⚠ et={et} で spkpos 失敗: {e}")
            continue

        r_km = np.linalg.norm(pos)
        utc = spice.et2utc(et, "ISOC", 3)

        ets_valid.append(et)
        times_utc.append(utc)
        distances_km.append(r_km)

    df = pd.DataFrame({
        "time_utc": pd.to_datetime(times_utc),
        "et": ets_valid,
        "L_km": distances_km,
    })

    df["L_m"] = df["L_km"] * 1000.0
    df["L_AU"] = df["L_km"] / 1.495978707e8

    return df


# ================================================================
# プロット
# ================================================================
def plot_distance(df, out_prefix):
    if df.empty:
        print("⚠ データが空です。プロットをスキップします。")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_utc"], df["L_km"], marker="o", ms=3)
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Distance L [km]")
    ax.set_title("New Horizons – Target Distance")
    ax.grid(True, ls="--", alpha=0.4)
    fig.autofmt_xdate()

    png_path = f"{out_prefix}_distance.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"✅ 図を保存しました: {png_path}")


# ================================================================
# MAIN
# ================================================================
def main():

    parser = argparse.ArgumentParser(
        description="New Horizons SPICE カーネルから幾何（距離 L(t)）を計算するスクリプト"
    )
    parser.add_argument("--kernel_dir", default=".", help="SPICE カーネルディレクトリ")
    parser.add_argument("--target", default="486958", help="ターゲット天体 (例: 486958 Arrokoth)")
    parser.add_argument("--observer", default="-98", help="観測者 NAIF ID (New Horizons = -98)")
    parser.add_argument("--frame", default="J2000", help="参照フレーム")
    parser.add_argument("--abcorr", default="LT+S", help="光行差補正 (LT+S 推奨)")
    parser.add_argument("--start", default="2019-01-01T04:00:00")
    parser.add_argument("--stop",  default="2019-01-01T08:00:00")
    parser.add_argument("--step", type=float, default=60.0)
    parser.add_argument("--out", default="newhorizons_tfgr")

    args = parser.parse_args()

    print("=== SPICE カーネルのロード ===")
    kernel_dir = os.path.abspath(args.kernel_dir)
    if not os.path.isdir(kernel_dir):
        raise FileNotFoundError(f"kernel_dir が見つかりません: {kernel_dir}")

    loaded = load_kernels(kernel_dir)

    if not loaded:
        print("⚠ カーネルが読み込まれていません。パスや拡張子を確認してください。")

    try:
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

        print("\n=== 統計量 [km] ===")
        print(df["L_km"].describe())

        csv_path = f"{args.out}_distance.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 幾何データを保存しました: {csv_path}")

        plot_distance(df, args.out)

    finally:
        spice.kclear()
        print("SPICE カーネルをアンロードしました。")


if __name__ == "__main__":
    main()
