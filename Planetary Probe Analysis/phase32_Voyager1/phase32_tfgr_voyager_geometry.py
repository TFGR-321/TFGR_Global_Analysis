import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError


def load_kernels(kernel_dir: str):
    """
    指定ディレクトリ中の SPICE カーネルを一括ロード
    （.bsp, .bc, .tpc, .tls, .tf, .tsc など）
    """
    loaded = []
    for fname in sorted(os.listdir(kernel_dir)):
        lower = fname.lower()
        if lower.endswith((".bsp", ".bc", ".tpc", ".tls", ".tf", ".tsc")):
            path = os.path.join(kernel_dir, fname)
            spice.furnsh(path)
            loaded.append(fname)
    return loaded


def compute_geometry(sc_id: int,
                     start_utc: str,
                     stop_utc: str,
                     step_sec: float,
                     observer: str = "JUPITER BARYCENTER"):
    """
    Voyager 宇宙機と木星系重心（デフォルト）との距離 L(t) を計算
    """
    # UTC→ET
    et_start = spice.str2et(start_utc)
    et_stop  = spice.str2et(stop_utc)

    ets = np.arange(et_start, et_stop + step_sec, step_sec)

    rows = []
    for et in ets:
        try:
            # target: 宇宙機, observer: Jupiter Barycenter
            pos, lt = spice.spkpos(str(sc_id), et, "J2000", "LT+S", observer)
            r_km = np.linalg.norm(pos)
        except SpiceyError as e:
            print("⚠ spkpos 失敗:", e)
            continue

        # 時刻を UTC 文字列に
        utc = spice.et2utc(et, "ISOC", 3)

        rows.append(
            {
                "time_utc": utc,
                "et": et,
                "L_km": r_km,
                "L_m": r_km * 1000.0,
                "L_AU": r_km / 149597870.7,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["time_utc", "et", "L_km", "L_m", "L_AU"])

    df = pd.DataFrame(rows)
    # time_utc を pandas の datetime に
    df["time_utc"] = pd.to_datetime(df["time_utc"])

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Voyager Jupiter encounter geometry for TFGR."
    )
    parser.add_argument(
        "--kernel_dir",
        required=True,
        help="Voyager 用 SPICE カーネルを入れたディレクトリ"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="開始時刻 UTC (例: 1979-03-05T00:00:00)"
    )
    parser.add_argument(
        "--stop",
        required=True,
        help="終了時刻 UTC (例: 1979-03-05T18:00:00)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=60.0,
        help="サンプリング間隔（秒）"
    )
    parser.add_argument(
        "--sc",
        choices=["vg1", "vg2"],
        default="vg1",
        help="宇宙機選択: vg1=Voyager 1, vg2=Voyager 2"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="出力ファイルのベース名（拡張子なし）"
    )
    parser.add_argument(
        "--observer",
        default="JUPITER BARYCENTER",
        help="観測点（デフォルト: JUPITER BARYCENTER）"
    )

    args = parser.parse_args()

    kernel_dir = args.kernel_dir
    if not os.path.isdir(kernel_dir):
        raise FileNotFoundError(f"kernel_dir が見つかりません: {kernel_dir}")

    # 宇宙機 ID を決定
    sc_id = -31 if args.sc == "vg1" else -32

    print("=== SPICE カーネルのロード ===")
    loaded = load_kernels(kernel_dir)
    print(f"✅ ロードしたカーネル数: {len(loaded)}")
    for f in loaded:
        print(f"   - {f}")

    try:
        print("\n=== 幾何計算 ===")
        df = compute_geometry(
            sc_id=sc_id,
            start_utc=args.start,
            stop_utc=args.stop,
            step_sec=args.step,
            observer=args.observer,
        )

        print("\n=== 計算結果プレビュー ===")
        print(df.head())

        print("\n=== 距離統計量 [km] ===")
        print(df["L_km"].describe())

        # 出力 CSV
        out_csv = f"{args.out}_distance.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n✅ 幾何データを保存しました: {out_csv}")

        # 図: 距離 vs 時刻
        if len(df) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(df["time_utc"], df["L_km"])
            plt.xlabel("Time [UTC]")
            plt.ylabel("Range L [km]")
            plt.title(f"Voyager ({args.sc}) – {args.observer} distance")
            plt.grid(True)
            out_png = f"{args.out}_distance.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"✅ 図を保存しました: {out_png}")
        else:
            print("⚠ データが空なのでプロットをスキップします。")

    finally:
        spice.kclear()
        print("SPICE カーネルをアンロードしました。")


if __name__ == "__main__":
    main()
