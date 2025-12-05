# digitize_gce_40x40.py
# Daylan+2014 Fig.6 左パネル（40x40°）のスペクトルを
# クリックして数値化する簡易 digitizer

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv

# ==== 1. 画像ファイル名（左パネルだけを切り出した PNG を用意してください） ====
IMG_FILE = "gce_fig6_left.png"  # ← 自分で保存した名前に合わせて変更

# ==== 2. 軸キャリブレーション（あなたが測ってくれた値） ====
#   x = a_x * log10(E[GeV]) + b_x
x_E1  = 227  # Eγ = 1 GeV
x_E10 = 431  # Eγ = 10 GeV

a_x = x_E10 - x_E1           # 1 decade 分のピクセル
b_x = x_E1                   # log10(1 GeV) = 0

def x_to_E(x):
    logE = (x - b_x) / a_x   # log10 E
    return 10.0**logE

#   y 軸は 1e-6 の位置だけを基準にして「相対スケール」で扱う
y_I1e6 = 258   # 1×10^-6 の水平線
# 便宜上、「1ピクセル上に行くと少し明るくなる」という線形近似をとる
# （絶対値より“スペクトルの形”が重要なので、ここは後で全体を正規化してもOK）
S_LOGL_PER_PIXEL = -0.01   # 傾き（適当な値→後で正規化）

def y_to_I(y):
    # y が小さいほど上（強度が大きい）
    dlog = (y - y_I1e6) * S_LOGL_PER_PIXEL
    logI = np.log10(1e-6) + dlog
    return 10.0**logI

# ==== 3. 画像を表示してクリックでデータ点を拾う ====
img = np.array(Image.open(IMG_FILE))

fig, ax = plt.subplots(figsize=(6,4))
ax.imshow(img)
ax.set_title("左図 40x40°: データ点を順番にクリックして、最後に右クリックまたは Enter で終了")
plt.axis("off")

print("データ点（中央の白丸）を順番にクリックしてください。")
print("終わったら右クリックするか、ウィンドウを閉じてください。")

clicked = plt.ginput(n=-1, timeout=0)  # 無制限にクリックを受け付ける
plt.close(fig)

xs = np.array([p[0] for p in clicked])
ys = np.array([p[1] for p in clicked])

Es  = x_to_E(xs)
Is  = y_to_I(ys)

# 強度は後で自由に正規化できるので、一度 E^2 dN/dE を相対値として扱う
# ここではそのまま I を E^2 dN/dE とみなす
E2dNdE = Is

# ==== 4. CSV に保存 ====
out_file = "gce_40x40_digitized.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["E_GeV", "E2dNdE_rel"])
    for E, val in zip(Es, E2dNdE):
        writer.writerow([E, val])

print(f"保存しました: {out_file}")
print("中身は相対強度なので、TFGR フィットのときに全体の正規化は自由に調整できます。")
