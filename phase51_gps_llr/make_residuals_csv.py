import pandas as pd
import os

base = r"C:\Users\PC2FW08_U\Desktop\GPS_SDTFT_Pipeline"
stations = ["AJAC", "ANK2", "ALIC", "MIZU"]

for st in stations:
    ztd_path = os.path.join(base, f"{st}_ztd_series.csv")
    slant_path = os.path.join(base, f"{st}_synthetic_slant_with_gmf.csv")

    if not os.path.exists(ztd_path):
        print(f"⚠️ {ztd_path} が見つかりません。スキップします。")
        continue
    if not os.path.exists(slant_path):
        print(f"⚠️ {slant_path} が見つかりません。スキップします。")
        continue

    print(f"\n📂 {st} データ読込中...")

    ztd = pd.read_csv(ztd_path)
    slant = pd.read_csv(slant_path)

    # ------------------------
    # 時刻列を柔軟に特定
    # ------------------------
    def find_time_col(df):
        for c in df.columns:
            if any(k in c.lower() for k in ["time", "utc", "epoch", "datetime", "date"]):
                return c
        raise ValueError("⚠️ 時刻列が見つかりません。列名を確認してください。")

    ztd_time_col = find_time_col(ztd)
    slant_time_col = find_time_col(slant)

    ztd.rename(columns={ztd_time_col: "time"}, inplace=True)
    slant.rename(columns={slant_time_col: "time"}, inplace=True)

    # ------------------------
    # 結合して残差計算
    # ------------------------
    merged = pd.merge(slant, ztd, on="time", how="left")

    if "TRODRY_m" not in merged.columns or "TROWET_m" not in merged.columns:
        print(f"⚠️ {st} で TRODRY_m/TROWET_m が見つかりません。スキップします。")
        continue

    merged["expected_slant_m"] = merged["TRODRY_m"] * merged["m_dry"] + merged["TROWET_m"] * merged["m_wet"]
    merged["residual_m"] = merged["slant_total_m"] - merged["expected_slant_m"]
    merged["station"] = st

    # 出力列
    keep = ["time", "sat", "elev_deg", "az_deg", "slant_total_m", "expected_slant_m", "residual_m", "station"]
    merged = merged[keep]

    out_path = os.path.join(base, f"{st}_residuals_lowElev_corrected.csv")
    merged.to_csv(out_path, index=False)
    print(f"✅ 出力完了: {out_path}")

print("\n🎯 すべての局で処理完了（またはスキップされました）")
