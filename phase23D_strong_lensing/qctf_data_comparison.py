# =============================================================
# QCTF vs Observational Data Comparison (Strong/Weak Lensing)
# Author: 三井貴浩
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load observational results ===========================
try:
    df_sl = pd.read_csv("stronglens_sigma_sweep_summary.csv")
    df_wl = pd.read_csv("wl_fit_summary.csv")
except Exception as e:
    print("❌ データファイルの読み込みに失敗しました:", e)
    exit()

print("列名（Strong lensing）:", list(df_sl.columns))
print("列名（Weak lensing）:", list(df_wl.columns))

# === 2. 自動列名判定 =========================================
def auto_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None

cLc_sl = auto_col(df_sl, ["best_log10_Lc", "log10Lc", "log10_lc"])
cp_sl  = auto_col(df_sl, ["best_p", "p"])
cq_sl  = auto_col(df_sl, ["best_q", "q"])

cLc_wl = auto_col(df_wl, ["best_log10_Lc", "theta_c", "log10Lc", "log10_lc"])
cp_wl  = auto_col(df_wl, ["best_p", "p"])
cq_wl  = auto_col(df_wl, ["best_q", "q"])

print("→ Strong lensing:", cLc_sl, cp_sl, cq_sl)
print("→ Weak lensing  :", cLc_wl, cp_wl, cq_wl)

# === 3. Extract representative best-fit parameters ============
p_sl, q_sl, Lc_sl = df_sl[cp_sl].mean(), df_sl[cq_sl].mean(), 10**df_sl[cLc_sl].mean()
p_wl, q_wl = df_wl[cp_wl].mean(), df_wl[cq_wl].mean()

# Lc (弱重力レンズ) の場合は θ_c [arcmin] → m に変換（おおよそ D_s=1Gpc 仮定）
theta_arcmin = df_wl[cLc_wl].mean()
Lc_wl = 1e9 * 3.086e16 * np.deg2rad(theta_arcmin / 60)

print("=== Strong Lensing (SL) ===")
print(f"  p = {p_sl:.3f}, q = {q_sl:.3f}, Lc = {Lc_sl:.2e}")
print("=== Weak Lensing (WL) ===")
print(f"  p = {p_wl:.3f}, q = {q_wl:.3f}, Lc = {Lc_wl:.2e}")

# === 4. Define QCTF flow ====================================
def beta_phi(phi, p, q):
    return q * p * (1.0 - np.exp(-phi / q))

def phi_flow(L_range, Lc, p, q):
    L0 = L_range[0]
    s = np.log(L_range / L0)
    sc = np.log(Lc / L0)
    phi = np.zeros_like(s)
    phi[0] = q * np.log(1 + np.exp(p * (s[0] - sc)))
    for i in range(len(s)-1):
        ds = s[i+1] - s[i]
        phi[i+1] = phi[i] + ds * beta_phi(phi[i], p, q)
    return s, phi

# === 5. Generate theoretical curves ==========================
L_range = np.logspace(3, 26, 400)
s_sl, phi_sl = phi_flow(L_range, Lc_sl, p_sl, q_sl)
s_wl, phi_wl = phi_flow(L_range, Lc_wl, p_wl, q_wl)

# === 6. Visualization =========================================
plt.figure(figsize=(8,6))
plt.plot(np.log10(L_range), phi_sl, label=f"Strong Lensing (p={p_sl:.2f}, q={q_sl:.2f})", color="blue", lw=2)
plt.plot(np.log10(L_range), phi_wl, label=f"Weak Lensing (p={p_wl:.2f}, q={q_wl:.2f})", color="orange", lw=2)
plt.xlabel(r"$\log_{10} L$  [m]", fontsize=12)
plt.ylabel(r"$\phi = \ln(\Delta t / \Delta t_0)$", fontsize=12)
plt.title("QCTF Flow Comparison: Strong vs Weak Lensing", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("QCTF_StrongWeakLensing_Flow.png", dpi=300)
plt.show()
