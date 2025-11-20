#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase38_tfgr_mass_scaling.py

Phase 38:
  L_cQ ∝ 1/m_a 仮説を組み込んだ
  階層ベイズ TFGR モデルの試験。

前提:
  - 入力は phase36_multiscale_dataset.csv 形式
      L_macro_m, L_int_m, dt_res_s, dt_err_s, pair
  - pair 名から使用原子種 (Sr, Yb, In) を推定し、
    有効質量 m_eff [atomic mass unit] を割り当てる。
  - 量子側臨界長を
      Lc_Q(pair) = A / m_eff(pair)
    として実装し、グローバル比例係数 A を推定する。

出力:
  - waic_mass_scaling.txt   : 質量スケーリングモデルの WAIC
  - trace_mass_scaling.txt  : 主要パラメータの事後サマリ
"""

import os
import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


# ------------------------------------------
# 1. ペア名 → 有効質量 m_eff のマッピング
# ------------------------------------------
def infer_mass_from_pair(pair_name):
    """
    非厳密だが、pair 名に含まれる文字列から有効質量 [u] を推定する。
    異種ペアの場合は平均質量。
    """
    # 代表的原子質量 [u]
    m_Sr = 88.0
    m_Yb = 171.0
    m_In = 115.0

    species = []
    s = pair_name.lower()
    if "sr" in s:
        species.append(m_Sr)
    if "yb" in s:
        species.append(m_Yb)
    if "in_" in s or "_in" in s:
        species.append(m_In)

    if not species:
        # デフォルト: Sr と同じにしておく（あくまで暫定）
        return m_Sr
    elif len(species) == 1:
        return species[0]
    else:
        # 複数種 → 平均質量
        return float(np.mean(species))


# ------------------------------------------
# 2. 質量スケーリング付き TFGR モデル
# ------------------------------------------
def build_mass_scaled_tfgr_model(
    Lm_scaled,
    Li_scaled,
    y_scaled,
    yerr_scaled,
    pair_idx,
    m_eff_scaled,   # 各ペアの有効質量（正規化済み）
    n_pair,
    draws,
    tune,
    chains,
    seed,
):
    """
    L_cQ(pair) = A / m_eff(pair) という制約付きの TFGR モデルを構築。
    ここでは Lc_Q(pair) 自体に「スケール空間でのスカラー A」をフィットする。

    L_macro 部分は Phase 37 と同様。
    """

    with pm.Model() as model:
        # ----- 階層バイアス -----
        b_pair = pm.Normal("b_pair", mu=0.0, sigma=2.0, shape=n_pair)
        sigma_pair = pm.HalfNormal("sigma_pair", sigma=1.0, shape=n_pair)

        # ----- マクロ側 TFGR -----
        dt0_M = pm.Normal("dt0_M", mu=0.0, sigma=2.0)
        log_Lc_M = pm.Normal("log_Lc_M", mu=np.log(1.0), sigma=1.0)  # ~ 1e8 m
        Lc_M = pm.Deterministic("Lc_M", pm.math.exp(log_Lc_M))
        p_M = pm.Normal("p_M", mu=0.5, sigma=0.5)
        q_M = pm.Normal("q_M", mu=1.5, sigma=0.7)

        # ----- 量子側 TFGR (質量スケーリング) -----
        # 基準質量 m0 を 100 u 付近とし、A は Lc_Q(m0) を与えるパラメータとする
        m0 = 100.0
        log_LcQ0 = pm.Normal("log_LcQ0", mu=np.log(3e-2), sigma=1.0)  # ~ 3e-5 m
        LcQ0 = pm.Deterministic("LcQ0", pm.math.exp(log_LcQ0))

        # 各ペアに対して Lc_Q(pair) = LcQ0 * (m0 / m_eff)
        # ここでは m_eff_scaled = m_eff / m0 なので、1/m_eff_scaled = m0 / m_eff
        inv_mass_factor = 1.0 / m_eff_scaled  # ~ m0 / m_eff
        Lc_Q_pair = pm.Deterministic("Lc_Q_pair", LcQ0 * inv_mass_factor[pair_idx])

        dt0_Q = pm.Normal("dt0_Q", mu=0.0, sigma=2.0)
        p_Q = pm.Normal("p_Q", mu=1.2, sigma=0.7)
        q_Q = pm.Normal("q_Q", mu=0.3, sigma=0.5)

        # TFGR 二階層項
        term_M = dt0_M * (1.0 + (Lm_scaled / Lc_M) ** p_M) ** q_M
        term_Q = dt0_Q * (1.0 + (Li_scaled / Lc_Q_pair) ** p_Q) ** q_Q
        tfgr_total = term_M + term_Q

        mu = b_pair[pair_idx] + tfgr_total
        sigma_tot = pm.math.sqrt(yerr_scaled ** 2 + sigma_pair[pair_idx] ** 2)

        pm.Normal("obs", mu=mu, sigma=sigma_tot, observed=y_scaled)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 2),
            target_accept=0.97,
            random_seed=seed,
            return_inferencedata=True,
        )

        idata.extend(pm.compute_log_likelihood(model=model, idata=idata))

    return idata


def extract_waic_values(waic_obj):
    if hasattr(waic_obj, "elpd_waic"):
        val = waic_obj.elpd_waic
        se = getattr(waic_obj, "se", getattr(waic_obj, "elpd_waic_se", np.nan))
        return val, se
    elif hasattr(waic_obj, "waic"):
        return waic_obj.waic, waic_obj.waic_se
    else:
        raise AttributeError("Unsupported ArviZ WAIC object.")


# ------------------------------------------
# 3. メイン
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 38: TFGR mass-scaling test (Lc_Q ∝ 1/m_eff)."
    )
    parser.add_argument("--csv", required=True,
                        help="phase36_multiscale_dataset.csv")
    parser.add_argument("--out", default="output_phase38_mass_scaling",
                        help="output directory")
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ----- データ読み込み -----
    df = pd.read_csv(args.csv)
    required = {"L_macro_m", "L_int_m", "dt_res_s", "dt_err_s", "pair"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV に必要な列 {required} の一部がありません。")

    Lm = df["L_macro_m"].values.astype(float)
    Li = df["L_int_m"].values.astype(float)
    y = df["dt_res_s"].values.astype(float)
    yerr = df["dt_err_s"].values.astype(float)
    pairs = df["pair"].values

    # pair -> index
    pair_labels = sorted(np.unique(pairs))
    pair_to_idx = {p: i for i, p in enumerate(pair_labels)}
    pair_idx = np.array([pair_to_idx[p] for p in pairs], dtype=int)
    n_pair = len(pair_labels)

    # ----- 有効質量 m_eff を計算 -----
    m_eff = np.array([infer_mass_from_pair(p) for p in pair_labels])  # [u]
    m0 = 100.0
    m_eff_scaled = m_eff / m0

    # ----- スケーリング -----
    L_macro_scale = 1.0e8
    L_int_scale = 1.0e-3
    y_scale = np.std(y) if np.std(y) > 0 else 1.0

    Lm_scaled = Lm / L_macro_scale
    Li_scaled = Li / L_int_scale
    y_scaled = y / y_scale
    yerr_scaled = yerr / y_scale

    # ----- モデルフィット -----
    print("Sampling mass-scaled TFGR2 model...")
    idata_ms = build_mass_scaled_tfgr_model(
        Lm_scaled, Li_scaled,
        y_scaled, yerr_scaled,
        pair_idx, m_eff_scaled,
        n_pair,
        draws=args.draws, tune=args.tune,
        chains=args.chains, seed=args.seed,
    )

    # ----- WAIC 計算 -----
    print("Computing WAIC for mass-scaled model...")
    waic_ms = az.waic(idata_ms)
    val_ms, se_ms = extract_waic_values(waic_ms)

    out_waic = os.path.join(args.out, "waic_mass_scaling.txt")
    with open(out_waic, "w", encoding="utf-8") as f:
        f.write("# Phase 38: TFGR mass-scaling model (Lc_Q ∝ 1/m_eff)\n\n")
        f.write(f"WAIC (mass-scaled TFGR2) = {val_ms:.3f} ± {se_ms:.3f}\n\n")
        f.write("Posterior medians:\n")
        post = idata_ms.posterior
        LcQ0_med = float(np.median(post["LcQ0"].values)) * L_int_scale
        dt0Q_med = float(np.median(post["dt0_Q"].values)) * y_scale
        pQ_med = float(np.median(post["p_Q"].values))
        qQ_med = float(np.median(post["q_Q"].values))
        f.write(f"  LcQ0 (m0=100u) = {LcQ0_med:.3e} [m]\n")
        f.write(f"  dt0_Q          = {dt0Q_med:.3e} [s]\n")
        f.write(f"  p_Q            = {pQ_med:.3f}\n")
        f.write(f"  q_Q            = {qQ_med:.3f}\n\n")
        f.write("Effective masses per pair:\n")
        for lab, m in zip(pair_labels, m_eff):
            f.write(f"  {lab}: m_eff = {m:.1f} u\n")

    print(f"✅ WAIC & parameter summary written to {out_waic}")


if __name__ == "__main__":
    main()
