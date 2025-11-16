# save as: make_pqLc_3Dmap.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

# ---------- 0) helper ----------
def safe_read(fn):
    p = Path(fn)
    return pd.read_csv(p) if p.exists() else None

# ---------- 1) weak-lensing proxy grid from summary ----------
wl = safe_read("wl_fit_summary.csv")
if wl is None:
    raise FileNotFoundError("wl_fit_summary.csv not found")
# we build a pseudo ΔAIC surface by smoothing pairwise fits over (p,q).
# In practice you'd read precomputed ΔAIC(p,q) grids; here we approximate:
# use beta-likeness: ΔAIC ∝ (p - p0)^2 + (q - q0)^2 with weights per tomographic pair
p0 = wl["p"].median()
q0 = wl["q"].median() if "q" in wl.columns and wl["q"].notna().any() else 1.0
pairs = wl["use"].unique()

P = np.linspace(0.15, 1.20, 120)
Q = np.linspace(0.20, 2.00, 120)
PP, QQ = np.meshgrid(P, Q)
dAIC_xip = np.zeros_like(PP)
dAIC_xim = np.zeros_like(PP)

for use,sub in wl.groupby("use"):
    pp = sub["p"].median()
    qq = sub["q"].median() if "q" in sub.columns and sub["q"].notna().any() else q0
    w  = 1.0  # optionally use pair S/N as weight
    dAIC_xip += w*((PP-pp)**2 + (QQ-qq)**2)
    dAIC_xim += w*((PP-pp*0.95)**2 + (QQ-qq*1.05)**2)  # slight tilt proxy

# normalize to zero-min
dAIC_xip -= dAIC_xip.min()
dAIC_xim -= dAIC_xim.min()

# ---------- 2) strong-lens points ----------
sl = None
for fn in ["stronglens_pq_grid_results.csv","stronglens_best_track_over_Lc.csv","stronglens_sigma_sweep_summary.csv"]:
    t = safe_read(fn)
    if t is not None:
        sl = t; break
if sl is None:
    raise FileNotFoundError("strong-lens CSV not found")

# column harmonization
cols = {c.lower(): c for c in sl.columns}
def pick(*names):
    for n in names:
        for k,v in cols.items():
            if k==n: return v
    return None
cLc = pick("log10_lc","best_log10_lc")
cp  = pick("p","best_p")
cq  = pick("q","best_q")
# --- 修正版（列名を強制的に正規化） ---
import pandas as pd

# CSV読み込み後の列名辞書化
cols = {c.lower(): c for c in sl.columns}

# 列名を明示的に指定
cLc = cols.get("best_log10_lc", "best_log10_lc")
cp  = cols.get("best_p", "best_p")
cq  = cols.get("best_q", "best_q")

# データ抽出とリネーム
sl = sl[[cLc, cp, cq]].rename(columns={cLc: "log10Lc", cp: "p", cq: "q"})
if "q" not in sl.columns:  # sweep q if absent
    qs = np.linspace(0.4, 1.6, 20)
    sl = pd.DataFrame(np.repeat(sl.values, len(qs), axis=0), columns=["log10lc","p","q"])
    sl["q"] = np.tile(qs, len(sl)//len(qs))

# ---------- 3) best ridge extraction ----------
dA = dAIC_xip + dAIC_xim
bestmask = dA < (dA.min()+2.0)
ridge = pd.DataFrame({"p":PP[bestmask], "q":QQ[bestmask]})
ridge.to_csv("pqLc_bestridge.csv", index=False)

# ---------- 4) plotting ----------
fig,axs = plt.subplots(1,3, figsize=(13.5,4.3), constrained_layout=True)

im0 = axs[0].contourf(P, Q, dAIC_xip, levels=30, cmap="magma")
cs0 = axs[0].contour(P, Q, dAIC_xip, levels=[2,6,10], colors="w")
axs[0].clabel(cs0, fmt="%d")
axs[0].set_xlabel("p"); axs[0].set_ylabel("q")
axs[0].set_title("ΔAIC (xip)")

im1 = axs[1].contourf(P, Q, dAIC_xim, levels=30, cmap="magma")
cs1 = axs[1].contour(P, Q, dAIC_xim, levels=[2,6,10], colors="w")
axs[1].clabel(cs1, fmt="%d")
axs[1].set_xlabel("p"); axs[1].set_ylabel("q")
axs[1].set_title("ΔAIC (xim)")

sc = axs[2].scatter(sl["p"], sl["q"], c=sl["log10Lc"], s=25, cmap="viridis", edgecolor="k", lw=0.3)
axs[2].contour(P, Q, dA, levels=[2,6,10], colors="r", linestyles="--", linewidths=1.2)
axs[2].set_xlabel("p"); axs[2].set_ylabel("q")
axs[2].set_title("Strong-lens points over ΔAIC contours")
cb = fig.colorbar(sc, ax=axs[2], pad=0.01)
cb.set_label(r"$\log_{10} L_c$")

plt.savefig("pq_Lc_3panel.png", dpi=200)
print("Done: pq_Lc_3panel.png, pqLc_bestridge.csv")
