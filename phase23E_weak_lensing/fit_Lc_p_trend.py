# save as: fit_Lc_p_trend.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor, RidgeCV

# ---------- 1) load ----------
def try_load_stronglens():
    cands = [
        "stronglens_best_track_over_Lc.csv",
        "stronglens_sigma_sweep_summary.csv",
        "stronglens_fineLc_track.csv"
    ]
    for fn in cands:
        p = Path(fn)
        if p.exists():
            df = pd.read_csv(p)
            # normalize column names
            cols = {c.lower(): c for c in df.columns}
            def pick(*names):
                for n in names:
                    if n in cols: return cols[n]
                return None
            cLc = pick("log10_lc","best_log10_lc","log10_lc[m]","log10_lc_m")
            cp  = pick("p","best_p")
            if cLc is None or cp is None:
                continue
            out = df[[cLc,cp]].rename(columns={cLc:"log10Lc",cp:"p"}).dropna()
            out["dataset"]="SL"
            return out
    raise FileNotFoundError("stronglens CSV not found")

def load_weaklens():
    p = Path("wl_fit_summary.csv")
    if not p.exists():
        raise FileNotFoundError("wl_fit_summary.csv not found")
    df = pd.read_csv(p)
    # expect columns: use, theta_c, p, q
    out = df[["theta_c","p"]].dropna().copy()
    # theta_c in arcmin -> define effective Lc by angular–diameter scaling proxy.
    # Here we map log10Lc := log10( k * theta_c[arcmin] ), k is absorbed in intercept.
    out["log10Lc"] = np.log10(out["theta_c"])
    out = out[["log10Lc","p"]]
    out["dataset"]="WL"
    return out

SL = try_load_stronglens()
WL = load_weaklens()
df = pd.concat([SL,WL], ignore_index=True)
df = df.replace([np.inf,-np.inf], np.nan).dropna()
X = df[["log10Lc"]].values
y = df["p"].values

# ---------- 2) models ----------
# M1: linear  p = a + b * log10Lc
lin = LinearRegression().fit(X,y)
y_lin = lin.predict(X)

# M2: quadratic p = a + b x + c x^2
X2 = np.hstack([X, X**2])
lin2 = LinearRegression().fit(X2,y)
y_quad = lin2.predict(X2)

# M3: log model p = a + b * ln(10)*log10Lc  (== linear in log but we compare as alternative)
X_log = np.log(10.0)*X
lin3 = LinearRegression().fit(X_log,y)
y_log = lin3.predict(X_log)

# Robust and ridge (for residual diagnostics)
huber = HuberRegressor().fit(X, y)
y_huber = huber.predict(X)
ridge = RidgeCV(alphas=np.logspace(-4,3,40)).fit(X, y)
y_ridge = ridge.predict(X)

# ---------- 3) information criteria ----------
def aic(y, yhat, k):
    n = len(y)
    rss = np.sum((y-yhat)**2)
    sigma2 = rss/n
    return n*np.log(sigma2) + 2*k

res = []
res.append(("linear", 2, aic(y,y_lin,2), lin.coef_[0], lin.intercept_))
res.append(("quadratic", 3, aic(y,y_quad,3), lin2.coef_[0], lin2.intercept_))
res.append(("log", 2, aic(y,y_log,2), lin3.coef_[0], lin3.intercept_))
tab = pd.DataFrame(res, columns=["model","k","AIC","coef1","intercept"])
tab["dAIC"] = tab["AIC"] - tab["AIC"].min()
tab.to_csv("Lc_p_trend_models.csv", index=False)

# ---------- 4) plot ----------
xx = np.linspace(X.min()*0.95, X.max()*1.05, 200)
yy_lin  = lin.predict(xx.reshape(-1,1))
yy_quad = lin2.predict(np.hstack([xx.reshape(-1,1), (xx**2).reshape(-1,1)]))
yy_log  = lin3.predict((np.log(10.0)*xx).reshape(-1,1))

plt.figure(figsize=(7.2,4.6))
ms = dict(SL=40, WL=30)
for g,sub in df.groupby("dataset"):
    plt.scatter(sub["log10Lc"], sub["p"], s=ms[g], alpha=0.85, label=g)
plt.plot(xx, yy_lin,  label=f"linear  (ΔAIC={tab.loc[tab.model=='linear','dAIC'].values[0]:.1f})", lw=2.2)
plt.plot(xx, yy_quad, label=f"quadratic(ΔAIC={tab.loc[tab.model=='quadratic','dAIC'].values[0]:.1f})", lw=2.2)
plt.plot(xx, yy_log,  label=f"log     (ΔAIC={tab.loc[tab.model=='log','dAIC'].values[0]:.1f})", lw=2.2)
plt.xlabel(r"$\log_{10} L_c$  (arbitrary offset absorbed)")
plt.ylabel(r"$p$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Lc_p_trend_fit.png", dpi=200)

# residuals (linear vs robust vs ridge)
plt.figure(figsize=(7.2,3.6))
plt.plot(df["log10Lc"], y - y_lin,  "o", ms=4, label="linear")
plt.plot(df["log10Lc"], y - y_huber,"o", ms=4, label="robust (Huber)")
plt.plot(df["log10Lc"], y - y_ridge,"o", ms=4, label="ridge")
plt.axhline(0,color="k",lw=1)
plt.xlabel(r"$\log_{10} L_c$")
plt.ylabel("residual p")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Lc_p_trend_residuals.png", dpi=200)
print("Done: Lc_p_trend_fit.png, Lc_p_trend_models.csv, Lc_p_trend_residuals.png")
