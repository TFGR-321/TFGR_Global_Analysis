import pandas as pd

df = pd.read_csv("phase42_gps_only_non_gps.csv")

# sat の1文字目で GNSS 系統判定 (C=BeiDou, E=Galileo, R=GLONASS, J=QZSS)
df["sys"] = df["sat"].astype(str).str[0]

for sys in ["C", "E", "R", "J"]:
    sub = df[df["sys"] == sys].drop(columns=["sys"])
    out = f"phase44_sys_{sys}.csv"
    sub.to_csv(out, index=False)
    print(sys, len(sub), "rows  ->", out)
