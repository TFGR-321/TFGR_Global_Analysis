#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase37_tfgr_hierarchical_bayes.py

Phase 37:
  二階層 TFGR モデルの「階層ベイズ版」フィット。

  - 入力: phase36_multiscale_dataset.csv 形式
      L_macro_m, L_int_m, dt_res_s, dt_err_s, pair
  - 各 pair ごとにバイアス b_pair と追加ノイズ sigma_pair を持つ階層モデル
  - dt0_M, Lc_M, p_M, q_M, dt0_Q, Lc_Q, p_Q, q_Q はグローバル共有
  - Baseline (GRのみ) と TFGR2 (二階層) を PyMC で MCMC フィットし、
    WAIC で比較する。

出力:
  - phase37_trace_summary.txt
  - waic_summary_phase37.txt
  - fit_Lmacro_vs_dt_phase37.png
  - fit_Lint_vs_dt_phase37.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az


# --------------------------------------------------
# ユーティリティ
# --------------------------------------------------
def extract_waic_values(waic_obj):
    """ArviZ WAIC オブジェクトから (value, se) を取り出す。"""
    if hasattr(waic_obj, "elpd_waic"):
        val = waic_obj.elpd_waic
        se = getattr(waic_obj, "se", getattr(waic_obj, "elpd_waic_se", np.nan))
        return val, se
    elif hasattr(waic_obj, "waic"):
        return waic_obj.waic, waic_obj.waic_se
    else:
        raise AttributeError("Unsupported ArviZ WAIC object.")


def tfgr2_term(Lm_scaled, Li_scaled,
               dt0_M, Lc_M, p_M, q_M,
               dt0_Q, Lc_Q, p_Q, q_Q):
    """スケール空間での二階層 TFGR 項 (次元なし)。"""
    term_M = dt0_M * (1.0 + (Lm_scaled / Lc_M) ** p_M) ** q_M
    term_Q = dt0_Q * (1.0 + (Li_scaled / Lc_Q) ** p_Q) ** q_Q
    return term_M + term_Q


# --------------------------------------------------
# Baseline モデル (GRのみ) - 階層
# --------------------------------------------------
def build_baseline_model(Lm_scaled, Li_scaled,
                         y_scaled, yerr_scaled,
                         pair_idx, n_pair,
                         draws, tune, chains, seed):
    """
    階層 Baseline モデル:
      Δt_scaled ~ Normal(b_pair[p], sqrt(yerr^2 + sigma_pair^2))
    """
    with pm.Model() as model:
        # pair ごとのオフセット
        b_pair = pm.Normal("b_pair", mu=0.0, sigma=2.0, shape=n_pair)
        # pair ごとの追加ノイズ (>0)
        sigma_pair = pm.HalfNormal("sigma_pair", sigma=1.0, shape=n_pair)

        mu = b_pair[pair_idx]
        sigma_tot = pm.math.sqrt(yerr_scaled ** 2 + sigma_pair[pair_idx] ** 2)

        pm.Normal("obs", mu=mu, sigma=sigma_tot, observed=y_scaled)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 2),
            target_accept=0.95,
            random_seed=seed,
            return_inferencedata=True,
        )

        # log_likelihood を追加（WAIC用）
        idata.extend(pm.compute_log_likelihood(model=model, idata=idata))

    return idata


# --------------------------------------------------
# TFGR2 モデル (二階層 TFGR + 階層バイアス)
# --------------------------------------------------
def build_tfgr2_model(Lm_scaled, Li_scaled,
                      y_scaled, yerr_scaled,
                      pair_idx, n_pair,
                      draws, tune, chains, seed,
                      LcM_prior=1.0, LcQ_prior=1e-2):
    """
    Δt_scaled = b_pair[p] + TFGR2(L_macro, L_int) + noise
    ここで L_macro, L_int はあらかじめスケール済み。
    LcM_prior, LcQ_prior は log-space の事前平均 (スケール空間での初期値)。
    """
    with pm.Model() as model:
        # ---- 階層バイアス & ノイズ ----
        b_pair = pm.Normal("b_pair", mu=0.0, sigma=2.0, shape=n_pair)
        sigma_pair = pm.HalfNormal("sigma_pair", sigma=1.0, shape=n_pair)

        # ---- マクロ側 TFGR パラメータ ----
        dt0_M = pm.Normal("dt0_M", mu=0.0, sigma=2.0)
        log_Lc_M = pm.Normal("log_Lc_M", mu=np.log(LcM_prior), sigma=1.0)
        Lc_M = pm.Deterministic("Lc_M", pm.math.exp(log_Lc_M))
        p_M = pm.Normal("p_M", mu=0.5, sigma=0.5)
        q_M = pm.Normal("q_M", mu=1.5, sigma=0.7)

        # ---- 量子側 TFGR パラメータ ----
        dt0_Q = pm.Normal("dt0_Q", mu=0.0, sigma=2.0)
        log_Lc_Q = pm.Normal("log_Lc_Q", mu=np.log(LcQ_prior), sigma=1.0)
        Lc_Q = pm.Deterministic("Lc_Q", pm.math.exp(log_Lc_Q))
        p_Q = pm.Normal("p_Q", mu=1.5, sigma=0.7)
        q_Q = pm.Normal("q_Q", mu=0.5, sigma=0.5)

        tfgr = tfgr2_term(Lm_scaled, Li_scaled,
                          dt0_M, Lc_M, p_M, q_M,
                          dt0_Q, Lc_Q, p_Q, q_Q)

        mu = b_pair[pair_idx] + tfgr
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


# --------------------------------------------------
# プロット（posterior median を使用）
# --------------------------------------------------
def plot_fits(df, idata_base, idata_tfgr,
              L_macro_scale, L_int_scale, y_scale,
              out_macro_png, out_int_png):
    Lm = df["L_macro_m"].values
    Li = df["L_int_m"].values
    y = df["dt_res_s"].values
    yerr = df["dt_err_s"].values

    # posterior median を取得
    post = idata_tfgr.posterior
    dt0_M_med = np.median(post["dt0_M"].values)
    Lc_M_med = np.median(post["Lc_M"].values)
    p_M_med = np.median(post["p_M"].values)
    q_M_med = np.median(post["q_M"].values)
    dt0_Q_med = np.median(post["dt0_Q"].values)
    Lc_Q_med = np.median(post["Lc_Q"].values)
    p_Q_med = np.median(post["p_Q"].values)
    q_Q_med = np.median(post["q_Q"].values)

    # ---- L_macro プロット ----
    Lm_plot = np.logspace(np.log10(Lm.min()), np.log10(Lm.max()), 200)
    Li_mean_scaled = (Li / L_int_scale).mean()
    Lm_plot_scaled = Lm_plot / L_macro_scale
    Li_plot_scaled = np.full_like(Lm_plot_scaled, Li_mean_scaled)

    tfgr_scaled_macro = tfgr2_term(
        Lm_plot_scaled, Li_plot_scaled,
        dt0_M_med, Lc_M_med, p_M_med, q_M_med,
        dt0_Q_med, Lc_Q_med, p_Q_med, q_Q_med,
    )
    tfgr_macro = tfgr_scaled_macro * y_scale

    # baseline は単なる pair オフセットの平均で近似的に一本線を描く
    b_pair_med = np.median(idata_base.posterior["b_pair"].values, axis=(0, 1))
    baseline_level = b_pair_med.mean() * y_scale

    plt.figure(figsize=(6, 4))
    plt.errorbar(Lm, y, yerr=yerr, fmt="o", alpha=0.5, label="data")
    plt.plot(Lm_plot, np.full_like(Lm_plot, baseline_level),
             "--", label="baseline")
    plt.plot(Lm_plot, tfgr_macro,
             "-", label=f"TFGR2 median (Lc_M={Lc_M_med*L_macro_scale:.1e} m)")
    plt.xscale("log")
    plt.xlabel("L_macro [m]")
    plt.ylabel("Residual Δt [s]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_macro_png, dpi=300)
    plt.close()

    # ---- L_int プロット ----
    Li_plot = np.logspace(np.log10(Li.min()), np.log10(Li.max()), 200)
    Lm_mean_scaled = (Lm / L_macro_scale).mean()
    Lm_plot_scaled2 = np.full_like(Li_plot, Lm_mean_scaled)
    Li_plot_scaled2 = Li_plot / L_int_scale

    tfgr_scaled_int = tfgr2_term(
        Lm_plot_scaled2, Li_plot_scaled2,
        dt0_M_med, Lc_M_med, p_M_med, q_M_med,
        dt0_Q_med, Lc_Q_med, p_Q_med, q_Q_med,
    )
    tfgr_int = tfgr_scaled_int * y_scale

    plt.figure(figsize=(6, 4))
    plt.errorbar(Li, y, yerr=yerr, fmt="o", alpha=0.5, label="data")
    plt.plot(Li_plot, np.full_like(Li_plot, baseline_level),
             "--", label="baseline")
    plt.plot(Li_plot, tfgr_int,
             "-", label=f"TFGR2 median (Lc_Q={Lc_Q_med*L_int_scale:.1e} m)")
    plt.xscale("log")
    plt.xlabel("L_int [m]")
    plt.ylabel("Residual Δt [s]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_int_png, dpi=300)
    plt.close()


# --------------------------------------------------
# メイン
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 37: Hierarchical Bayesian TFGR2 fit."
    )
    parser.add_argument("--csv", required=True,
                        help="multiscale dataset CSV (phase36_multiscale_dataset.csv)")
    parser.add_argument("--out", default="output_phase37",
                        help="output directory")
    parser.add_argument("--draws", type=int, default=500,
                        help="MCMC draws (per chain)")
    parser.add_argument("--tune", type=int, default=500,
                        help="MCMC tune (per chain)")
    parser.add_argument("--chains", type=int, default=2,
                        help="number of MCMC chains")
    parser.add_argument("--seed", type=int, default=123,
                        help="random seed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- データ読み込み ----
    df = pd.read_csv(args.csv)
    required = {"L_macro_m", "L_int_m", "dt_res_s", "dt_err_s", "pair"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV に必要な列 {required} の一部がありません。")

    Lm = df["L_macro_m"].values.astype(float)
    Li = df["L_int_m"].values.astype(float)
    y = df["dt_res_s"].values.astype(float)
    yerr = df["dt_err_s"].values.astype(float)

    # pair -> index
    pair_labels = sorted(df["pair"].unique())
    pair_to_idx = {p: i for i, p in enumerate(pair_labels)}
    pair_idx = df["pair"].map(pair_to_idx).values
    n_pair = len(pair_labels)

    # ---- スケーリング ----
    L_macro_scale = 1.0e8   # 地球規模スケール
    L_int_scale = 1.0e-3    # mm スケール
    y_scale = np.std(y) if np.std(y) > 0 else 1.0

    Lm_scaled = Lm / L_macro_scale
    Li_scaled = Li / L_int_scale
    y_scaled = y / y_scale
    yerr_scaled = yerr / y_scale

    # ---- Baseline ----
    print("Sampling hierarchical Baseline model...")
    idata_base = build_baseline_model(
        Lm_scaled, Li_scaled,
        y_scaled, yerr_scaled,
        pair_idx, n_pair,
        draws=args.draws, tune=args.tune,
        chains=args.chains, seed=args.seed,
    )

    # ---- TFGR2 ----
    print("\nSampling hierarchical TFGR2 model...")
    # Phase 36 の結果を踏まえた log-Lc 事前平均 (スケール空間)
    LcM_prior_scaled = 1.0   # => 1e8 m
    LcQ_prior_scaled = 3.0e-2  # => 3e-5 m (L_int_scale=1e-3 を掛ける)
    idata_tfgr = build_tfgr2_model(
        Lm_scaled, Li_scaled,
        y_scaled, yerr_scaled,
        pair_idx, n_pair,
        draws=args.draws, tune=args.tune,
        chains=args.chains, seed=args.seed,
        LcM_prior=LcM_prior_scaled,
        LcQ_prior=LcQ_prior_scaled,
    )

    # ---- WAIC ----
    print("\nComputing WAIC...")
    waic_base = az.waic(idata_base)
    waic_tfgr = az.waic(idata_tfgr)
    base_val, base_se = extract_waic_values(waic_base)
    tfgr_val, tfgr_se = extract_waic_values(waic_tfgr)
    delta_waic = tfgr_val - base_val  # TFGR - Baseline

    out_waic = os.path.join(args.out, "waic_summary_phase37.txt")
    with open(out_waic, "w", encoding="utf-8") as f:
        f.write("# Phase 37 Hierarchical Bayesian TFGR2 fit\n\n")
        f.write("Baseline (hierarchical GR-only):\n")
        f.write(f"  WAIC = {base_val:.3f} ± {base_se:.3f}\n\n")
        f.write("TFGR2 (hierarchical, two-scale):\n")
        f.write(f"  WAIC = {tfgr_val:.3f} ± {tfgr_se:.3f}\n\n")
        f.write("Comparison (smaller is better):\n")
        f.write(f"  ΔWAIC (TFGR2 - Baseline) = {delta_waic:.3f}\n")
        f.write("  → ΔWAIC < 0 なら TFGR2 モデルの方が好ましい。\n\n")

        post = idata_tfgr.posterior
        dt0_M_med = np.median(post["dt0_M"].values) * y_scale
        Lc_M_med = np.median(post["Lc_M"].values) * L_macro_scale
        p_M_med = np.median(post["p_M"].values)
        q_M_med = np.median(post["q_M"].values)
        dt0_Q_med = np.median(post["dt0_Q"].values) * y_scale
        Lc_Q_med = np.median(post["Lc_Q"].values) * L_int_scale
        p_Q_med = np.median(post["p_Q"].values)
        q_Q_med = np.median(post["q_Q"].values)

        f.write("Posterior medians (physical units):\n")
        f.write(f"  dt0_M = {dt0_M_med:.3e} [s]\n")
        f.write(f"  Lc_M  = {Lc_M_med:.3e} [m]\n")
        f.write(f"  p_M   = {p_M_med:.3f}\n")
        f.write(f"  q_M   = {q_M_med:.3f}\n")
        f.write(f"  dt0_Q = {dt0_Q_med:.3e} [s]\n")
        f.write(f"  Lc_Q  = {Lc_Q_med:.3e} [m]\n")
        f.write(f"  p_Q   = {p_Q_med:.3f}\n")
        f.write(f"  q_Q   = {q_Q_med:.3f}\n")

    print(f"✅ WAIC summary written to {out_waic}")

    # ---- ざっくりトレースサマリ ----
    out_trace = os.path.join(args.out, "phase37_trace_summary.txt")
    with open(out_trace, "w", encoding="utf-8") as f:
        f.write(str(az.summary(idata_tfgr, var_names=[
            "dt0_M", "Lc_M", "p_M", "q_M",
            "dt0_Q", "Lc_Q", "p_Q", "q_Q"
        ])))
    print(f"✅ Trace summary written to {out_trace}")

    # ---- プロット ----
    out_macro_png = os.path.join(args.out, "fit_Lmacro_vs_dt_phase37.png")
    out_int_png = os.path.join(args.out, "fit_Lint_vs_dt_phase37.png")
    plot_fits(df, idata_base, idata_tfgr,
              L_macro_scale, L_int_scale, y_scale,
              out_macro_png, out_int_png)
    print(f"✅ Plots saved to {out_macro_png} and {out_int_png}")


if __name__ == "__main__":
    main()
