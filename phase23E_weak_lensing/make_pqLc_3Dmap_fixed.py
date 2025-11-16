#!/usr/bin/env python3
# ================================================================
# make_pqLc_3Dmap_fixed.py
# 強・弱重力レンズ解析統合用 3Dマップ生成スクリプト（修正版）
# ------------------------------------------------
# 対応: pandas 2.x / matplotlib 3.x / numpy 1.26+
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Δt(L) 統合マップ生成スクリプト ===")

# ------------------------------------------------
# 1. CSV ファイルの読み込み
# ------------------------------------------------
fname = "stronglens_sigma_sweep_summary.csv"
if not os.path.exists(fname):
    raise FileNotFoundError(f"❌ {fname} が見つかりません。")

sl = pd.read_csv(fname)
print(f"✅ 読み込み完了: {len(sl)} rows, {len(sl.columns)} columns")
print("列名一覧:", list(sl.columns))

# ------------------------------------------------
# 2. 列名の自動検出（大文字・小文字の違いを吸収）
# ------------------------------------------------
cols_lower = {c.lower(): c for c in sl.columns}

def get_col(name):
    """大文字小文字を無視して列名を取得"""
    for key, val in cols_lower.items():
        if name.lower() == key:
            return val
    return None

cLc = get_col("best_log10_Lc")
cp  = get_col("best_p")
cq  = get_col("best_q")

print("✅ 検出された列名:", cLc, cp, cq)

if None in [cLc, cp, cq]:
    print("❌ 必要な列が見つかりません。")
    print("既存の列:", list(sl.columns))
    raise SystemExit

# ------------------------------------------------
# 3. データ抽出とリネーム
# ------------------------------------------------
sl = sl[[cLc, cp, cq]].rename(columns={cLc: "log10Lc", cp: "p", cq: "q"})
print("✅ データ整形後:", sl.columns.tolist())

# ------------------------------------------------
# 4. 軸範囲とメッシュの設定
# ------------------------------------------------
Lc_vals = sl["log10Lc"].values
p_vals  = sl["p"].values
q_vals  = sl["q"].values

print(f"範囲: log10Lc=[{Lc_vals.min():.2f},{Lc_vals.max():.2f}], p=[{p_vals.min():.2f},{p_vals.max():.2f}], q=[{q_vals.min():.2f},{q_vals.max():.2f}]")

# ------------------------------------------------
# 5. 3D プロット
# ------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(Lc_vals, p_vals, q_vals, c=q_vals, cmap="plasma", s=40, edgecolor="k", alpha=0.8)

ax.set_xlabel(r"$\log_{10} L_c$")
ax.set_ylabel(r"$p$")
ax.set_zlabel(r"$q$")
ax.set_title("Strong-Lens Best-Fit Parameters")

plt.tight_layout()
plt.savefig("pq_Lc_3panel.png", dpi=300)
print("✅ 図を保存しました: pq_Lc_3panel.png")

# ------------------------------------------------
# 6. 最良フィットの統計出力
# ------------------------------------------------
best = sl.loc[sl["p"].idxmin()]
best.to_csv("pqLc_bestridge.csv", index=False)
print("✅ 最良点を保存しました: pqLc_bestridge.csv")

print("=== 完了しました！ ===")
