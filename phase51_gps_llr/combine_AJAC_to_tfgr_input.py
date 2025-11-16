#!/usr/bin/env python
# combine_AJAC_to_tfgr_input.py
# AJAC_synthetic_slant_with_gmf.csv + AJAC_ztd_series.csv → Phase 51-B 入力 CSV

import pandas as pd
import numpy as np

C_LIGHT = 299792458.0  # m/s
R_EARTH = 6371000.0    # m

# ① ファイル読み込み
slant = pd.read_csv("AJAC_synthetic_slant_with_gmf.csv")
ztd   = pd.read_csv("AJAC_ztd_series.csv")

# ② 仰角から「実効高度」と地心距離を近似計算
slant["elev_rad"] = np.deg2rad(slant["elev_deg"])
slant["H_m"]      = slant["slant_total_m"] * np.sin(slant["elev_rad"])
slant["L_m"]      = R_EARTH + slant["H_m"]

# ③ スラント遅延 → 時間残差 [s]
slant["dt_res_s"] = slant["slant_total_m"] / C_LIGHT

# ④ 時刻で ZTD 情報をマージ（最も近い時刻を採用）
ztd["time_utc"]   = pd.to_datetime(ztd["time_utc"])
slant["time_utc"] = pd.to_datetime(slant["time_utc_x"])

merged = pd.merge_asof(
    slant.sort_values("time_utc"),
    ztd.sort_values("time_utc"),
    on="time_utc",
    direction="nearest"
)

# ⑤ 出力列を整理
out_cols = [
    "time_utc", "station", "sat",
    "L_m", "dt_res_s",
    "TROTOT_m", "TRODRY_m", "TROWET_m"
]
out = merged[out_cols]

out.to_csv("AJAC_phase51B_tfgr_input.csv", index=False)
print("✅ 出力完了: AJAC_phase51B_tfgr_input.csv")
