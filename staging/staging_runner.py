"""
staging_runner.py — Staging bake-off for top 7 architectures.

Runs stratified 5-fold CV for each architecture with boosted convergence
budgets (n_estimators=2000, early_stopping=100), computes OOF ROC AUC,
generates test predictions (averaged across folds), and outputs a comparison table.

Usage:
    uv run python staging/staging_runner.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Parent dir for prepare.py imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier

from prepare import evaluate_predictions, load_splits, load_train_data

# ──────────────────────────────────────────────────────────────────────────────
# Global constants — boosted convergence (time is NOT a constraint)
# ──────────────────────────────────────────────────────────────────────────────
N_EST = 2000  # generous tree budget; early stopping picks the sweet spot
EARLY_STOP = 100  # patient stopping
N_JOBS = 8
SEED = 101

# ──────────────────────────────────────────────────────────────────────────────
# 7 Architecture Configs (all boosted with N_EST / EARLY_STOP)
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    # ── 1. Iter 10 — High-res bins, equal blend ──
    "high_res_bins": {
        "lgb": dict(
            num_leaves=40, learning_rate=0.1, max_depth=-1,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=6, learning_rate=0.085, max_bin=2048,
            eval_metric="logloss",
        ),
        "w_lgb": 0.50, "w_xgb": 0.50,
        "asymmetric": False,
    },
    # ── 2. Iter 11 — Ultra-high-res XGB, XGB-dominant ──
    "xgb_maxbin4096_w65": {
        "lgb": dict(
            num_leaves=40, learning_rate=0.1, max_depth=-1,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=6, learning_rate=0.085, max_bin=4096,
            eval_metric="logloss",
        ),
        "w_lgb": 0.35, "w_xgb": 0.65,
        "asymmetric": False,
    },
    # ── 3. Iter 12 — Regularised (depth-5 XGB + leaves-36 LGB) ──
    "regularized_lgb36_xgb_d5": {
        "lgb": dict(
            num_leaves=36, learning_rate=0.1, max_depth=6,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=5, learning_rate=0.085, max_bin=4096,
            eval_metric="logloss",
        ),
        "w_lgb": 0.35, "w_xgb": 0.65,
        "asymmetric": False,
    },
    # ── 4. Iter 12b — + colsample 0.95 / gamma 0.05 ──
    "lgb_cs95_xgb_gamma05": {
        "lgb": dict(
            num_leaves=36, learning_rate=0.1, max_depth=6,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.95,
        ),
        "xgb": dict(
            max_depth=5, learning_rate=0.085, max_bin=4096,
            gamma=0.05, eval_metric="logloss",
        ),
        "w_lgb": 0.35, "w_xgb": 0.65,
        "asymmetric": False,
    },
    # ── 5. Iter 13 — Asymmetric depth-5 (originally timeout 126s) ──
    "asymmetric_depth5_v1": {
        "lgb": dict(
            num_leaves=36, learning_rate=0.088, max_depth=6,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=5, learning_rate=0.092, max_bin=4096,
            eval_metric="auc",
        ),
        "w_lgb": 0.38, "w_xgb": 0.62,
        "asymmetric": True,
    },
    # ── 6. Iter 13b — Asymmetric fast variant ──
    "asymmetric_depth5_fast": {
        "lgb": dict(
            num_leaves=36, learning_rate=0.089, max_depth=6,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=5, learning_rate=0.092, max_bin=4096,
            eval_metric="logloss",
        ),
        "w_lgb": 0.38, "w_xgb": 0.62,
        "asymmetric": True,
    },
    # ── 7. Iter 14 — Final best (logloss eval + net_green + sub_x_inc) ──
    "asymmetric_depth5_v2_final": {
        "lgb": dict(
            num_leaves=36, learning_rate=0.088, max_depth=6,
            min_child_samples=30, max_bin=1024,
            subsample=0.8, colsample_bytree=0.8,
        ),
        "xgb": dict(
            max_depth=5, learning_rate=0.092, max_bin=4096,
            eval_metric="logloss",
        ),
        "w_lgb": 0.38, "w_xgb": 0.62,
        "asymmetric": True,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering (mirrors train.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
FREQ_COLS = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
CAT_COLS = [
    "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]


def preprocess(df: pd.DataFrame, group_stats: dict | None = None):
    """
    Feature engineering matching train.py.
    Computes group stats from df if group_stats is None (train),
    otherwise re-uses provided stats (test).
    Returns (X, extra_xgb_df, group_stats).
    """
    X = df.copy()

    # Domain features
    total_chg = X["Charging_Stations_Near_Home"] + X["Charging_Stations_Near_Work"]
    X["charging_stations_total"] = total_chg.astype(np.int32)
    X["charging_home_work_ratio"] = (
        X["Charging_Stations_Near_Home"] / (X["Charging_Stations_Near_Work"] + 1.0)
    ).astype(np.float32)
    X["commute_per_station"] = (X["Daily_Commute_km"] / (total_chg + 1.0)).astype(np.float32)
    X["income_per_car"] = (
        X["Annual_Income_USD"] / (X["Number_of_Cars_Owned"] + 1.0)
    ).astype(np.float32)

    # Ordinal maps
    anx = X["Range_Anxiety_Level"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(2).astype(np.int32)
    hc = X["Home_Charging_Possible"].map({"No": 0, "Yes": 1}).fillna(0).astype(np.int32)
    sub = X["Subsidy_Available"].map({"No": 0, "Yes": 1}).fillna(0).astype(np.int32)

    X["unmitigated_range_anxiety"] = (anx * (1 - hc)).astype(np.int32)
    X["green_subsidy_synergy"] = (X["Environmental_Concern_Level"] * sub).astype(np.float32)

    # Group aggregations
    stats = {}
    for grp in ["City_Type", "Current_Car_Type"]:
        if group_stats is None:
            inc_mean = X.groupby(grp, observed=False)["Annual_Income_USD"].mean()
            com_mean = X.groupby(grp, observed=False)["Daily_Commute_km"].mean()
            stats[grp] = {"inc": inc_mean, "com": com_mean}
        else:
            inc_mean = group_stats[grp]["inc"]
            com_mean = group_stats[grp]["com"]
        X[f"income_diff_{grp}"] = (X["Annual_Income_USD"] - X[grp].map(inc_mean)).fillna(0).astype(np.float32)
        X[f"commute_diff_{grp}"] = (X["Daily_Commute_km"] - X[grp].map(com_mean)).fillna(0).astype(np.float32)
    if group_stats is None:
        group_stats = stats

    # Asymmetric XGB-only features (computed for all; only used when config says so)
    env_f = X["Environmental_Concern_Level"].astype(np.float32)
    anx_f = anx.astype(np.float32)
    sub_f = sub.astype(np.float32)
    inc_f = X["Annual_Income_USD"].astype(np.float32)
    extra_xgb = pd.DataFrame(
        {
            "net_green_subsidy": (sub_f * (env_f - anx_f)).astype(np.float32),
            "sub_x_inc": (sub_f * (inc_f / 10000.0)).astype(np.float32),
        },
        index=X.index,
    )

    # Categoricals
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Downcast
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype(np.float32)
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype(np.int32)

    return X, extra_xgb, group_stats


# ──────────────────────────────────────────────────────────────────────────────
# Single-architecture runner
# ──────────────────────────────────────────────────────────────────────────────
def run_arch(
    name: str,
    cfg: dict,
    X_all: pd.DataFrame,
    y: pd.Series,
    splits: list,
    xgb_extra_all: pd.DataFrame,
    X_test: pd.DataFrame,
    xgb_extra_test: pd.DataFrame,
):
    print(f"\n{'='*70}")
    print(f"  ARCH: {name}")
    print(f"  Blend: LGB {cfg['w_lgb']:.2f} / XGB {cfg['w_xgb']:.2f}  |  Asymmetric: {cfg['asymmetric']}")
    print(f"{'='*70}")
    t0 = time.time()

    n_tr, n_te = len(y), len(X_test)
    oof = np.zeros(n_tr, dtype=np.float64)
    test_lgb = np.zeros(n_te, dtype=np.float64)
    test_xgb = np.zeros(n_te, dtype=np.float64)
    w_l, w_x = cfg["w_lgb"], cfg["w_xgb"]
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        ft = time.time()
        Xtr = X_all.iloc[tr_idx].copy()
        ytr = y.iloc[tr_idx]
        Xva = X_all.iloc[va_idx].copy()
        yva = y.iloc[va_idx]
        Xte = X_test.copy()

        # In-fold frequency encoding
        for col in FREQ_COLS:
            fm = Xtr[col].value_counts(normalize=True)
            Xtr[f"freq_{col}"] = Xtr[col].map(fm).fillna(0).astype(np.float32)
            Xva[f"freq_{col}"] = Xva[col].map(fm).fillna(0).astype(np.float32)
            Xte[f"freq_{col}"] = X_test[col].map(fm).fillna(0).astype(np.float32)

        # ── LightGBM ──
        m_lgb = lgb.LGBMClassifier(
            n_estimators=N_EST, n_jobs=N_JOBS,
            random_state=SEED + fold, verbose=-1,
            **cfg["lgb"],
        )
        m_lgb.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False)],
        )
        p_lgb_va = m_lgb.predict_proba(Xva)[:, 1].astype(np.float64)
        p_lgb_te = m_lgb.predict_proba(Xte)[:, 1].astype(np.float64)
        lgb_it = m_lgb.best_iteration_
        del m_lgb; gc.collect()

        # ── Asymmetric features for XGB ──
        if cfg["asymmetric"]:
            for c in xgb_extra_all.columns:
                Xtr[c] = xgb_extra_all[c].iloc[tr_idx].values
                Xva[c] = xgb_extra_all[c].iloc[va_idx].values
                Xte[c] = xgb_extra_test[c].values

        # ── XGBoost ──
        m_xgb = XGBClassifier(
            n_estimators=N_EST, tree_method="hist",
            enable_categorical=True, n_jobs=N_JOBS,
            random_state=SEED + fold,
            early_stopping_rounds=EARLY_STOP,
            **cfg["xgb"],
        )
        m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p_xgb_va = m_xgb.predict_proba(Xva)[:, 1].astype(np.float64)
        p_xgb_te = m_xgb.predict_proba(Xte)[:, 1].astype(np.float64)
        xgb_it = m_xgb.best_iteration
        del m_xgb; gc.collect()

        # Rank blend OOF
        r_l = rankdata(p_lgb_va) / len(p_lgb_va)
        r_x = rankdata(p_xgb_va) / len(p_xgb_va)
        oof[va_idx] = w_l * r_l + w_x * r_x

        # Accumulate test (average raw probs across folds, rank blend at end)
        test_lgb += p_lgb_te / 5.0
        test_xgb += p_xgb_te / 5.0

        fs = evaluate_predictions(yva, oof[va_idx])
        fold_scores.append(fs)
        print(f"  Fold {fold}: AUC={fs:.6f}  LGB_iters={lgb_it}  XGB_iters={xgb_it}  ({time.time()-ft:.1f}s)")

        del Xtr, ytr, Xva, yva, Xte; gc.collect()

    oof_auc = evaluate_predictions(y, oof)

    # Rank blend averaged test predictions
    test_preds = w_l * (rankdata(test_lgb) / n_te) + w_x * (rankdata(test_xgb) / n_te)

    elapsed = time.time() - t0
    print(f"  ────────────────────────────────────")
    print(f"  OOF ROC AUC = {oof_auc:.6f}  |  {elapsed:.1f}s total")
    return oof_auc, oof, test_preds, fold_scores, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STAGE = os.path.join(BASE, "staging")
    os.makedirs(STAGE, exist_ok=True)

    print("=" * 70)
    print("  STAGING MODEL BAKE-OFF")
    print(f"  n_estimators={N_EST}  early_stopping={EARLY_STOP}  seed={SEED}")
    print(f"  Architectures: {len(CONFIGS)}")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data …")
    X_raw, y = load_train_data()
    splits = load_splits()
    test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
    test_ids = test_df["id"].values
    test_feat = test_df.drop(columns=["id"])
    del test_df; gc.collect()
    print(f"  Train {X_raw.shape[0]:,} x {X_raw.shape[1]}  |  Test {test_feat.shape[0]:,} x {test_feat.shape[1]}")

    # Preprocess (shared)
    print("[2/3] Preprocessing …")
    X_all, xgb_extra, g_stats = preprocess(X_raw)
    X_test, xgb_extra_test, _ = preprocess(test_feat, group_stats=g_stats)
    del X_raw, test_feat; gc.collect()
    print(f"  Features: {X_all.shape[1]} train cols, {X_test.shape[1]} test cols")

    # Run architectures
    print("[3/3] Running bake-off …")
    rows = []
    for name, cfg in CONFIGS.items():
        auc, oof, tpreds, fscores, elapsed = run_arch(
            name, cfg, X_all, y, splits, xgb_extra, X_test, xgb_extra_test,
        )
        # Save artefacts
        np.save(os.path.join(STAGE, f"{name}_oof_preds.npy"), oof)
        sub = pd.DataFrame({"id": test_ids, "Will_Buy_EV": tpreds})
        sub.to_csv(os.path.join(STAGE, f"{name}_submission.csv"), index=False)
        print(f"  → saved {name}_submission.csv")

        rows.append(dict(
            architecture=name, oof_roc_auc=round(auc, 6),
            fold_0=round(fscores[0], 6), fold_1=round(fscores[1], 6),
            fold_2=round(fscores[2], 6), fold_3=round(fscores[3], 6),
            fold_4=round(fscores[4], 6), runtime_s=round(elapsed, 1),
        ))
        gc.collect()

    # Comparison table
    df = pd.DataFrame(rows).sort_values("oof_roc_auc", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    df.to_csv(os.path.join(STAGE, "comparison_table.csv"))

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON TABLE (sorted by OOF ROC AUC)")
    print(f"{'='*70}")
    print(df.to_string())
    print(f"\n  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"  Best → {df.iloc[0]['architecture']}  OOF AUC = {df.iloc[0]['oof_roc_auc']:.6f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
