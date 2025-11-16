import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Phase 22 — TFGR-based BTFR (M_b vs V0)
# ==========================================

# 1. 銀河名と TFGR パラメータ（Phase 19 の結果から）
names  = np.array(["NGC 2403", "NGC 3198", "NGC 2903", "NGC 5055"])

V0     = np.array([108.33, 119.47, 174.73, 152.34])   # [km/s]
V0_err = np.array([  0.43,   0.73,   1.18,   0.72])   # [km/s]

# 2. バリオン質量 M_b [10^10 M_sun]
#    ★★ ここをあなたの値に書き換えてください ★★
#    （下の値は「だいたいこんな感じ」の仮置きです）
Mbar_1e10 = np.array([
    0.9,   # NGC 2403 (例)
    2.3,   # NGC 3198 (例)
    4.0,   # NGC 2903 (例)
    7.0    # NGC 5055 (例)
])
Mbar_err_1e10 = np.array([
    0.1,   # 見積もり誤差（仮）: ±0.1×10^10 M_sun など
    0.2,
    0.4,
    0.7
])

# 3. 単位をそろえる
#    M_b [M_sun] に変換
Mbar   = Mbar_1e10 * 1e10
Mbar_err = Mbar_err_1e10 * 1e10

# 4. log10 に変換
logV  = np.log10(V0)
# 誤差伝播: sigma(log10 V) = sigma(V) / (V ln 10)
logV_err = V0_err / (V0 * np.log(10.0))

logM  = np.log10(Mbar)
logM_err = Mbar_err / (Mbar * np.log(10.0))

# 5. 直線フィット: logM = a * logV + b
#    y = a x + b を最小二乗（y誤差を重み）でフィット
w = 1.0 / (logM_err**2)   # 重み = 1/σ^2
X = np.vstack([logV, np.ones_like(logV)])  # [2, N]
# 重み付き最小二乗
WX = X * w
WY = logM * w
# (a, b) = (WX X^T)^(-1) WX Y
A = WX @ X.T
B = WX @ logM
params = np.linalg.solve(A, B)
a, b = params[0], params[1]

# 共分散行列（2パラ）
cov = np.linalg.inv(A)
a_err = np.sqrt(cov[0,0])
b_err = np.sqrt(cov[1,1])

print("\n===== TFGR-based BTFR Fit =====")
print("Relation: log10(M_b/M_sun) = a * log10(V0 / km s^-1) + b")
print(f"a = {a:.3f} ± {a_err:.3f}")
print(f"b = {b:.3f} ± {b_err:.3f}")
print("================================\n")

# 6. プロット
plt.figure(figsize=(7,6))

# データ点
for i, name in enumerate(names):
    plt.errorbar(logV[i], logM[i],
                 xerr=logV_err[i], yerr=logM_err[i],
                 fmt="o", capsize=3, label=name)

# フィット直線
x_fit = np.linspace(min(logV)-0.1, max(logV)+0.1, 200)
y_fit = a * x_fit + b
plt.plot(x_fit, y_fit, "k-", label=f"Fit: a={a:.2f} ± {a_err:.2f}")

# 参考として「傾き4」の線も描く（通常のBTFR）
# 中心はフィット直線と同じ logV の中央値付近に合わせる
x0 = np.mean(logV)
y0 = a * x0 + b
y_fit4 = 4.0 * (x_fit - x0) + y0
plt.plot(x_fit, y_fit4, "k--", alpha=0.4, label="Slope 4 (classic BTFR)")

plt.xlabel(r"$\log_{10}(V_0 / \mathrm{km\ s^{-1}})$")
plt.ylabel(r"$\log_{10}(M_{\mathrm{bar}} / M_\odot)$")
plt.title("TFGR-based Baryonic Tully-Fisher Relation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
